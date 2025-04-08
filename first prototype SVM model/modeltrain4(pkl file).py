import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
import pickle
import sklearn
import time

# 打印 scikit-learn 的版本
#print(f"scikit-learn 版本: {sklearn.__version__}")

# 先通过交叉验证平均准确率来选择和优化模型，然后使用模型准确率来评估最终模型的性能。

# 设置随机种子
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

def extract_force_data_and_calculate_stats(folder_path):
    all_stats = []
    all_fft_data = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)
                try:
                    # 读取 CSV 文件
                    df = pd.read_csv(file_path)
                    # 提取第三列 Force 信号数据
                    force_data = df.iloc[:, 2].values
                    # 过滤负数
                    force_data = force_data[force_data >= 0]

                    if len(force_data) == 0:
                        print(f"文件 {file_path} 过滤后 force_data 为空，跳过此文件。")
                        continue

                    # 计算统计特征
                    mean_force = np.mean(force_data)
                    std_force = np.std(force_data)
                    # 力的上升速率
                    if len(force_data) > 1:
                        diff_force = np.diff(force_data)
                        if len(diff_force) > 0:
                            rising_rate = np.mean(
                                diff_force[diff_force > 0]) if np.sum(diff_force > 0) > 0 else 0
                        else:
                            rising_rate = 0
                    else:
                        rising_rate = 0
                    # 力的峰值
                    peak_force = np.max(force_data)

                    # 进行 FFT
                    N = len(force_data)
                    T = 0.1  # 假设采样周期为 1
                    yf = fft(force_data)
                    xf = fftfreq(N, T)[:N // 2]
                    yf_abs = 2.0 / N * np.abs(yf[:N // 2])

                    # 存储统计结果
                    stats = {
                        'file': file,
                        'mean_force': mean_force,
                        'std_force': std_force,
                        'rising_rate': rising_rate,
                        'peak_force': peak_force
                    }
                    all_stats.append(stats)
                    all_fft_data.append(yf_abs)

                except pd.errors.ParserError as e:
                    print(
                        f"解析 CSV 文件 {file_path} 时出错: {e}。请检查文件内容是否为有效的 CSV 格式。")
                except Exception as e:
                    print(
                        f"处理文件 {file_path} 时出错: {e}。请检查文件是否损坏或格式是否正确。")
    return all_stats, all_fft_data


def prepare_dataset(stats_folder1, fft_data_folder1, stats_folder2, fft_data_folder2):
    X = []
    y = []
    # 找到最大的 FFT 长度
    max_fft_length = max([len(fft_data)
                          for fft_data in fft_data_folder1 + fft_data_folder2])

    for stats, fft_data in zip(stats_folder1, fft_data_folder1):
        features = [stats['mean_force'], stats['std_force'],
                    stats['rising_rate'], stats['peak_force']]
        # 填充 FFT 结果到最大长度
        padded_fft_data = np.pad(
            fft_data, (0, max_fft_length - len(fft_data)), mode='constant')
        features.extend(padded_fft_data)
        X.append(features)
        y.append(0)  # 0 表示 ripe grape

    for stats, fft_data in zip(stats_folder2, fft_data_folder2):
        features = [stats['mean_force'], stats['std_force'],
                    stats['rising_rate'], stats['peak_force']]
        # 填充 FFT 结果到最大长度
        padded_fft_data = np.pad(
            fft_data, (0, max_fft_length - len(fft_data)), mode='constant')
        features.extend(padded_fft_data)
        X.append(features)
        y.append(1)  # 1 表示 rotten grape

    X = np.array(X)
    y = np.array(y)

    return X, y, max_fft_length  # 返回 max_fft_length


def predict_grape_type(force_data, max_fft_length, model, weights):
    # 过滤负数
    force_data = force_data[force_data >= 0]

    if len(force_data) == 0:
        print("输入的 force_data 过滤后为空，无法进行预测。")
        return None

    # 计算统计特征
    mean_force = np.mean(force_data)
    std_force = np.std(force_data)
    # 力的上升速率
    if len(force_data) > 1:
        diff_force = np.diff(force_data)
        if len(diff_force) > 0:
            rising_rate = np.mean(
                diff_force[diff_force > 0]) if np.sum(diff_force > 0) > 0 else 0
        else:
            rising_rate = 0
    else:
        rising_rate = 0
    peak_force = np.max(force_data)

    # 进行 FFT
    N = len(force_data)
    T = 0.1  # 采样频率为10/s采样周期T=1/f
    yf = fft(force_data)
    xf = fftfreq(N, T)[:N // 2]
    yf_abs = 2.0 / N * np.abs(yf[:N // 2])

    # 填充 FFT 结果到最大长度
    padded_fft_data = np.pad(
        yf_abs, (0, max_fft_length - len(yf_abs)), mode='constant')

    # 构建特征向量
    features = [mean_force, std_force, rising_rate, peak_force]
    features.extend(padded_fft_data)
    features = np.array(features)
    weighted_features = features * weights
    weighted_features = weighted_features.reshape(1, -1)

    # 进行预测
    prediction = model.predict(weighted_features)
    if prediction[0] == 0:
        return "ripe grape"
    else:
        return "rotten grape"


# 两个文件夹路径，使用原始字符串
folder1 = r'd:\大二下学期\cdio\ripe'
folder2 = r'd:\大二下学期\cdio\rotten'

# 处理第一个文件夹
stats_folder1, fft_data_folder1 = extract_force_data_and_calculate_stats(folder1)
# 处理第二个文件夹
stats_folder2, fft_data_folder2 = extract_force_data_and_calculate_stats(folder2)

# 准备数据集
X, y, max_fft_length = prepare_dataset(stats_folder1, fft_data_folder1,
                                       stats_folder2, fft_data_folder2)

# 数据集划分
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_SEED)

# 初始化权重
num_features = X_train.shape[1]
weights = np.ones(num_features)

# 加入正则化，使用 L2 正则化
model = SGDClassifier(penalty='l2', alpha=0.18, random_state=RANDOM_SEED)

# 记录开始时间
start_time = time.time()

# 自适应调整权重训练模型
learning_rate = 0.01
num_epochs = 100
losses = []  # 用于记录每个 epoch 的损失值

for epoch in range(num_epochs):
    total_loss = 0
    for i in range(len(X_train)):
        weighted_X_train = X_train[i] * weights
        weighted_X_train = weighted_X_train.reshape(1, -1)
        model.partial_fit(weighted_X_train, [y_train[i]], classes=[0, 1])
        # 简单的损失计算（这里可以根据实际情况修改）
        prediction = model.predict(weighted_X_train)
        loss = (prediction[0] - y_train[i]) ** 2
        total_loss += loss
        # 梯度更新权重
        gradient = 2 * (prediction[0] - y_train[i]) * X_train[i]
        weights = weights - learning_rate * gradient
    epoch_loss = total_loss / len(X_train)
    losses.append(epoch_loss)
 #   print(f"Epoch {epoch + 1}, Loss: {epoch_loss}")

# 记录结束时间
end_time = time.time()

# 计算训练时间
training_time = end_time - start_time
print(f"模型训练时间: {training_time} 秒")

# 加入交叉验证
cv_scores = cross_val_score(model, X_train, y_train, cv=5)
print(f"交叉验证平均准确率: {cv_scores.mean()}")

# 保存模型为 pkl 格式
model_filename = 'svm_model.pkl'
with open(model_filename, 'wb') as f:
    pickle.dump(model, f)
print(f"模型已保存到 {model_filename}")

# 保存权重为 pkl 格式
weights_filename = 'weights.pkl'
with open(weights_filename, 'wb') as f:
    pickle.dump(weights, f)
print(f"权重已保存到 {weights_filename}")

# 保存 max_fft_length 为 pkl 格式
max_fft_length_filename = 'max_fft_length.pkl'
with open(max_fft_length_filename, 'wb') as f:
    pickle.dump(max_fft_length, f)
print(f"max_fft_length 已保存到 {max_fft_length_filename}")

# 提取指定特征的权重
specific_weights = weights[:4]
# 计算权重比
total_weight = np.sum(specific_weights)
weight_ratios = specific_weights / total_weight

# 输出指定特征的权重比
print("mean force, std force, rising rate, peak force 的权重比：")
print(f"mean force: {weight_ratios[0]:.4f}")
print(f"std force: {weight_ratios[1]:.4f}")
print(f"rising rate: {weight_ratios[2]:.4f}")
print(f"peak force: {weight_ratios[3]:.4f}")

# 绘制损失曲线
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.plot(range(1, num_epochs + 1), losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss over Epochs')

# 预测
y_pred = []
for i in range(len(X_test)):
    weighted_X_test = X_test[i] * weights
    weighted_X_test = weighted_X_test.reshape(1, -1)
    y_pred.append(model.predict(weighted_X_test)[0])

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print(f"模型准确率: {accuracy}")

