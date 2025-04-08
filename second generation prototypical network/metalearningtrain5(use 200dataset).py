import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from sklearn.model_selection import train_test_split, StratifiedKFold
import torch
import torch.nn as nn
import torch.optim as optim
import copy
import pickle
import time

# 设置随机种子
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# 获取所有标签的函数 
def all_labels(y):
    return np.unique(y)

# --- 文件处理和特征提取函数 ---
def extract_force_data_and_calculate_stats(folder_path):
    all_stats = []
    all_fft_data = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)
                df = pd.read_csv(file_path)
                force_data = pd.to_numeric(df.iloc[:, 2]).values
                force_data = force_data[~np.isnan(force_data)]
                force_data = force_data[force_data >= 0]
                if len(force_data) < 2:
                    continue
                mean_force = np.mean(force_data)
                std_force = np.std(force_data)
                diff_force = np.diff(force_data)
                rising_rate = np.mean(diff_force[diff_force > 0]) if np.sum(diff_force > 0) > 0 else 0
                peak_force = np.max(force_data)
                N = len(force_data)
                T = 0.1
                yf = fft(force_data)
                yf_abs = 2.0 / N * np.abs(yf[:N // 2])
                stats = {
                    'file': file,
                    'mean_force': mean_force,
                    'std_force': std_force,
                    'rising_rate': rising_rate,
                    'peak_force': peak_force
                }
                all_stats.append(stats)
                all_fft_data.append(yf_abs)
    return all_stats, all_fft_data

def prepare_dataset(stats_folder1, fft_data_folder1, stats_folder2, fft_data_folder2):
    X = []
    y = []
    all_fft_data_combined = [fft for fft in fft_data_folder1 + fft_data_folder2 if len(fft) > 0]
    max_fft_length = max([len(fft_data) for fft_data in all_fft_data_combined])
    for stats, fft_data in zip(stats_folder1, fft_data_folder1):
        features = [stats['mean_force'], stats['std_force'],
                    stats['rising_rate'], stats['peak_force']]
        padded_fft_data = np.pad(
            fft_data, (0, max_fft_length - len(fft_data)), mode='constant', constant_values=0.0)
        features.extend(padded_fft_data)
        X.append(features)
        y.append(0)
    for stats, fft_data in zip(stats_folder2, fft_data_folder2):
        features = [stats['mean_force'], stats['std_force'],
                    stats['rising_rate'], stats['peak_force']]
        padded_fft_data = np.pad(
            fft_data, (0, max_fft_length - len(fft_data)), mode='constant', constant_values=0.0)
        features.extend(padded_fft_data)
        X.append(features)
        y.append(1)
    X = np.array(X, dtype=np.float64)
    y = np.array(y)
    return X, y, max_fft_length

# --- 预测函数 ---
def predict_grape_type(force_data, max_fft_length, model, device, prototypes):
    model.eval()
    force_data = np.array(force_data)
    force_data = force_data[force_data >= 0]
    if len(force_data) < 2:
        return None
    mean_force = np.mean(force_data)
    std_force = np.std(force_data)
    diff_force = np.diff(force_data)
    rising_rate = np.mean(diff_force[diff_force > 0]) if np.sum(diff_force > 0) > 0 else 0
    peak_force = np.max(force_data)
    N = len(force_data)
    T = 0.1
    yf = fft(force_data)
    yf_abs = 2.0 / N * np.abs(yf[:N // 2])
    current_fft_len = len(yf_abs)
    if current_fft_len > max_fft_length:
        padded_fft_data = yf_abs[:max_fft_length]
    elif current_fft_len < max_fft_length:
        padded_fft_data = np.pad(
            yf_abs, (0, max_fft_length - current_fft_len), mode='constant', constant_values=0.0)
    else:
        padded_fft_data = yf_abs
    features = [mean_force, std_force, rising_rate, peak_force]
    features.extend(padded_fft_data)
    features = np.array(features, dtype=np.float64)
    features = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        query_features = model(features)
        if not isinstance(prototypes, torch.Tensor):
            prototypes_tensor = torch.tensor(prototypes, dtype=torch.float32).to(device)
        else:
            prototypes_tensor = prototypes.to(device)
        distances = torch.cdist(query_features, prototypes_tensor)
        prediction = torch.argmin(distances, dim=1).item()
    return "ripe grape" if prediction == 0 else "rotten grape"

# --- 定义带 Dropout 的原型网络模型 ---
class PrototypicalNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0.5):
        super(PrototypicalNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# --- 计算原型函数 ---
def calculate_prototypes(X, y, model, device):
    model.eval()
    prototypes = []
    unique_classes = torch.unique(y)
    for c in unique_classes:
        class_indices = torch.where(y == c)[0]
        if len(class_indices) > 0:
            class_features = model(X[class_indices])
            prototype = torch.mean(class_features, dim=0)
            prototypes.append(prototype)
    prototypes = torch.stack(prototypes)
    return prototypes

# --- 主要执行流程 ---

# 1. 定义超参数
folder1 = r'd:\大二下学期\cdio\ripe'
folder2 = r'd:\大二下学期\cdio\rotten'
test_split_ratio = 0.2
validation_split_ratio = 0.2
hidden_size = 32
output_size = 16
dropout_rate = 0.5
learning_rate = 0.003
weight_decay_rate = 1e-3
num_epochs = 100
early_stopping_patience = 10
n_splits = 5

# 2. 加载和准备数据
stats_folder1, fft_data_folder1 = extract_force_data_and_calculate_stats(folder1)
stats_folder2, fft_data_folder2 = extract_force_data_and_calculate_stats(folder2)
X, y, max_fft_length = prepare_dataset(stats_folder1, fft_data_folder1,
                                       stats_folder2, fft_data_folder2)

# 3. 数据集划分
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=test_split_ratio, random_state=RANDOM_SEED, stratify=y)

# 交叉验证
skf = StratifiedKFold(n_splits=n_splits, random_state=RANDOM_SEED, shuffle=True)
cv_scores = []
fold_times = []

for fold, (train_index, val_index) in enumerate(skf.split(X_temp, y_temp)):
    X_train, X_val = X_temp[train_index], X_temp[val_index]
    y_train, y_val = y_temp[train_index], y_temp[val_index]

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size = X_train.shape[1]
    model = PrototypicalNetwork(input_size, hidden_size, output_size, dropout_rate).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay_rate)

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None

    start_time = time.time()
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        X_train_dev = X_train_tensor.to(device)
        y_train_dev = y_train_tensor.to(device)
        train_prototypes = calculate_prototypes(X_train_dev, y_train_dev, model, device)
        train_query_features = model(X_train_dev)
        train_distances = torch.cdist(train_query_features, train_prototypes)
        train_loss = criterion(-train_distances, y_train_dev)
        train_losses.append(train_loss.item())
        train_loss.backward()
        optimizer.step()

        if len(X_val) > 0:
            model.eval()
            with torch.no_grad():
                X_val_dev = X_val_tensor.to(device)
                y_val_dev = y_val_tensor.to(device)
                current_prototypes_for_eval = calculate_prototypes(X_train_dev, y_train_dev, model, device)
                val_query_features = model(X_val_dev)
                val_distances = torch.cdist(val_query_features, current_prototypes_for_eval)
                val_loss = criterion(-val_distances, y_val_dev)
                val_losses.append(val_loss.item())

            if val_loss.item() < best_val_loss:
                best_val_loss = val_loss.item()
                epochs_no_improve = 0
                best_model_state = copy.deepcopy(model.state_dict())
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= early_stopping_patience:
                break

    end_time = time.time()
    fold_time = end_time - start_time
    fold_times.append(fold_time)

    if len(X_val) > 0 and best_model_state:
        model.load_state_dict(best_model_state)



    if len(X_val) > 0:
        model.eval()
        with torch.no_grad():
            X_val_dev = X_val_tensor.to(device)
            y_val_dev = y_val_tensor.to(device)
            val_prototypes = calculate_prototypes(X_train_dev, y_train_dev, model, device)
            val_query_features = model(X_val_dev)
            val_distances = torch.cdist(val_query_features, val_prototypes)
            y_pred_val = torch.argmin(val_distances, dim=1).cpu().numpy()
            fold_accuracy = np.mean(y_pred_val == y_val)
            cv_scores.append(fold_accuracy)
            print(f"第 {fold + 1} 折验证集准确率: {fold_accuracy:.4f}")
            print(f"第 {fold + 1} 折训练时间: {fold_time:.4f} 秒")

    # 保存各折的最佳模型
    if best_model_state:
        model_path = f'model_fold_{fold + 1}.pth'
        torch.save(best_model_state, model_path)
        print(f"第 {fold + 1} 折的最佳模型已保存到 {model_path}")

print(f"交叉验证平均得分: {np.mean(cv_scores)}")
print(f"交叉验证平均训练时间: {np.mean(fold_times):.4f} 秒")

# 最后，使用全部训练数据（X_temp, y_temp）训练最终模型
X_train_tensor = torch.tensor(X_temp, dtype=torch.float32)
y_train_tensor = torch.tensor(y_temp, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_size = X_temp.shape[1]
model = PrototypicalNetwork(input_size, hidden_size, output_size, dropout_rate).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay_rate)

train_losses = []
val_losses = []
best_val_loss = float('inf')
epochs_no_improve = 0
best_model_state = None

start_time = time.time()
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    X_train_dev = X_train_tensor.to(device)
    y_train_dev = y_train_tensor.to(device)
    train_prototypes = calculate_prototypes(X_train_dev, y_train_dev, model, device)
    train_query_features = model(X_train_dev)
    train_distances = torch.cdist(train_query_features, train_prototypes)
    train_loss = criterion(-train_distances, y_train_dev)
    train_losses.append(train_loss.item())
    train_loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        X_test_dev = X_test_tensor.to(device)
        y_test_dev = y_test_tensor.to(device)
        test_prototypes = calculate_prototypes(X_train_dev, y_train_dev, model, device)
        test_query_features = model(X_test_dev)
        test_distances = torch.cdist(test_query_features, test_prototypes)
        y_pred_test = torch.argmin(test_distances, dim=1).cpu().numpy()
        test_accuracy = np.mean(y_pred_test == y_test)

end_time = time.time()
final_train_time = end_time - start_time

print(f"测试集准确率: {test_accuracy:.4f}")
print(f"最终模型训练时间: {final_train_time:.4f} 秒")

# 保存最终模型为pkl格式
final_model_path = 'best_prototypical_model.pkl'
with open(final_model_path, 'wb') as f:
    pickle.dump(model.state_dict(), f)
print(f"最终模型已保存到 {final_model_path}")

# 保存最大FFT长度为pkl格式
max_fft_length_path = 'metamax_fft_length.pkl'
with open(max_fft_length_path, 'wb') as f:
    pickle.dump(max_fft_length, f)
print(f"最大FFT长度已保存到 {max_fft_length_path}")

# 保存最终原型为pkl格式
final_prototypes_path = 'final_prototypes.pkl'
final_prototypes = calculate_prototypes(X_train_dev, y_train_dev, model, device)
with open(final_prototypes_path, 'wb') as f:
    pickle.dump(final_prototypes.detach().cpu().numpy(), f)
print(f"最终原型已保存到 {final_prototypes_path}")
    