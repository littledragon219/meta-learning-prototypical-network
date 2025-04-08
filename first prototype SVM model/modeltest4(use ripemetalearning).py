import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
import pickle

# 加载模型、权重和max_fft_length
model_filename = 'svm_model2.pkl'
weights_filename = 'weights2.pkl'
max_fft_length_filename = 'max_fft_length2.pkl'

with open(model_filename, 'rb') as f:
    model = pickle.load(f)
with open(weights_filename, 'rb') as f:
    weights = pickle.load(f)
with open(max_fft_length_filename, 'rb') as f:
    max_fft_length = pickle.load(f)

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
                    # 处理负数值
                    force_data[force_data < 0] = 0

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

def prepare_dataset(stats_folder1, fft_data_folder1, stats_folder2, fft_data_folder2, target_fft_length):
    X = []
    y = []
    
    for stats, fft_data in zip(stats_folder1, fft_data_folder1):
        features = [stats['mean_force'], stats['std_force'],
                   stats['rising_rate'], stats['peak_force']]
        # 截断或填充FFT数据到目标长度
        if len(fft_data) > target_fft_length:
            fft_data = fft_data[:target_fft_length]
        else:
            fft_data = np.pad(fft_data, (0, target_fft_length - len(fft_data)), mode='constant')
        features.extend(fft_data)
        X.append(features)
        y.append(0)

    for stats, fft_data in zip(stats_folder2, fft_data_folder2):
        features = [stats['mean_force'], stats['std_force'],
                   stats['rising_rate'], stats['peak_force']]
        if len(fft_data) > target_fft_length:
            fft_data = fft_data[:target_fft_length]
        else:
            fft_data = np.pad(fft_data, (0, target_fft_length - len(fft_data)), mode='constant')
        features.extend(fft_data)
        X.append(features)
        y.append(1)

    return np.array(X), np.array(y), target_fft_length

def predict_grape_type(force_data, max_fft_length, model, weights):
    # 过滤负数
    force_data = force_data[force_data >= 0]

    if len(force_data) == 0:
        print("输入的 force_data 过滤后为空，无法进行预测。")
        return None

    # 计算统计特征
    mean_force = np.mean(force_data)
    std_force = np.std(force_data)
    if len(force_data) > 1:
        diff_force = np.diff(force_data)
        if len(diff_force) > 0:
            rising_rate = np.mean(diff_force[diff_force > 0]) if np.sum(diff_force > 0) > 0 else 0
        else:
            rising_rate = 0
    else:
        rising_rate = 0
    peak_force = np.max(force_data)

    # 进行FFT
    N = len(force_data)
    T = 0.1
    yf = fft(force_data)
    xf = fftfreq(N, T)[:N // 2]
    yf_abs = 2.0 / N * np.abs(yf[:N // 2])

    # 截断或填充FFT数据到目标长度
    if len(yf_abs) > max_fft_length:
        yf_abs = yf_abs[:max_fft_length]
    else:
        yf_abs = np.pad(yf_abs, (0, max_fft_length - len(yf_abs)), mode='constant')

    # 构建特征向量
    features = [mean_force, std_force, rising_rate, peak_force]
    features.extend(yf_abs)
    features = np.array(features)
    weighted_features = features * weights
    weighted_features = weighted_features.reshape(1, -1)

    # 进行预测
    prediction = model.predict(weighted_features)
    if prediction[0] == 0:
        return "ripe grape"
    else:
        return "rotten grape"

# 加载模型、权重和max_fft_length
model_filename = 'svm_model2.pkl'
weights_filename = 'weights2.pkl'
max_fft_length_filename = 'max_fft_length2.pkl'

with open(model_filename, 'rb') as f:
    model = pickle.load(f)
with open(weights_filename, 'rb') as f:
    weights = pickle.load(f)
with open(max_fft_length_filename, 'rb') as f:
    max_fft_length = pickle.load(f)

# 处理测试数据
test_folder = r'd:\大二下学期\cdio\test'
ripe_folder = os.path.join(test_folder, 'ripe')
rotten_folder = os.path.join(test_folder, 'rotten')

# 处理测试数据
stats_ripe, fft_data_ripe = extract_force_data_and_calculate_stats(ripe_folder)
stats_rotten, fft_data_rotten = extract_force_data_and_calculate_stats(rotten_folder)

# 准备测试数据集，使用保存的max_fft_length
X_test, y_test, _ = prepare_dataset(stats_ripe, fft_data_ripe,
                                  stats_rotten, fft_data_rotten,
                                  target_fft_length=max_fft_length)

# 预测
y_pred = []
for i in range(len(X_test)):
    weighted_X_test = X_test[i] * weights
    weighted_X_test = weighted_X_test.reshape(1, -1)
    y_pred.append(model.predict(weighted_X_test)[0])

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print(f"模型准确率: {accuracy}")
   # 计算混淆矩阵
cm = confusion_matrix(y_test, y_pred)
print("混淆矩阵:")
print(cm)

   # 绘制混淆矩阵图
plt.figure(figsize=(6, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion matrix')
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ['Ripe', 'Rotten'])
plt.yticks(tick_marks, ['Ripe', 'Rotten'])

thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
           plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

   # 计算 ROC 曲线和 AUC
y_score = model.decision_function(X_test * weights)
fpr, tpr, thresholds = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)

   # 绘制 ROC 曲线
plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()
   