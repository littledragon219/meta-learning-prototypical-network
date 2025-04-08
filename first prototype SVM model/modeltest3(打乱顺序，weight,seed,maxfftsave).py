import os
import pandas as pd
import numpy as np
from scipy.fft import fft, fftfreq
from joblib import dump, load
from sklearn.utils import shuffle

# 设置随机种子
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


def extract_force_data_and_calculate_stats(file_path):
    try:
        df = pd.read_csv(file_path)
        force_data = df.iloc[:, 2].values
        force_data = force_data[force_data >= 0]
        if len(force_data) == 0:
            return None

        mean_force = np.mean(force_data)
        std_force = np.std(force_data)
        diff_force = np.diff(force_data)
        rising_rate = np.mean(diff_force[diff_force > 0]) if len(diff_force) > 0 and np.sum(diff_force > 0) > 0 else 0
        peak_force = np.max(force_data)

        N = len(force_data)
        T = 0.1
        yf = fft(force_data)
        xf = fftfreq(N, T)[:N // 2]
        yf_abs = 2.0 / N * np.abs(yf[:N // 2])

        return mean_force, std_force, rising_rate, peak_force, yf_abs
    except pd.errors.ParserError:
        return None
    except Exception:
        return None


def prepare_features(mean_force, std_force, rising_rate, peak_force, fft_data, max_fft_length):
    if max_fft_length < len(fft_data):
        padded_fft_data = fft_data[:max_fft_length]
    else:
        padded_fft_data = np.pad(fft_data, (0, max_fft_length - len(fft_data)), mode='constant')

    features = [mean_force, std_force, rising_rate, peak_force]
    features.extend(padded_fft_data)
    features = np.array(features)
    return features


def predict_grape_type(features, model, weights):
    if len(features) != len(weights):
        return None
    weighted_features = features * weights
    weighted_features = weighted_features.reshape(1, -1)
    prediction = model.predict(weighted_features)
    return 0 if prediction[0] == 0 else 1


model_filename = r'D:/大二下学期/CDIO/ripegrape/first prototype SVM model/svm_model.joblib'
if not os.path.exists(model_filename):
    print(f"错误: 模型文件 {model_filename} 不存在，请检查路径。")
else:
    model = load(model_filename)
    weights_filename = r'D:/大二下学期/CDIO/ripegrape/first prototype SVM model/weights.joblib'
    if not os.path.exists(weights_filename):
        print(f"错误: 权重文件 {weights_filename} 不存在，请检查路径。")
    else:
        weights = load(weights_filename)
        max_fft_length_filename = r'D:/大二下学期/CDIO/ripegrape/max_fft_length.joblib'
        if os.path.exists(max_fft_length_filename):
            max_fft_length = load(max_fft_length_filename)
        else:
            print(f"错误: 未找到 max_fft_length 文件，请确保你已经运行过训练代码并保存了该文件。")
            exit()

        # 两个新文件夹路径
        folder1 = r'd:\大二下学期\cdio\ripe3'
        folder2 = r'd:\大二下学期\cdio\rotten2'

        X = []
        y = []

        # 处理第一个文件夹（成熟葡萄）
        for root, dirs, files in os.walk(folder1):
            for file in files:
                if file.endswith('.csv'):
                    file_path = os.path.join(root, file)
                    result = extract_force_data_and_calculate_stats(file_path)
                    if result is not None:
                        mean_force, std_force, rising_rate, peak_force, fft_data = result
                        features = prepare_features(
                            mean_force, std_force, rising_rate, peak_force, fft_data, max_fft_length)
                        if features is not None:
                            X.append(features)
                            y.append(0)

        # 处理第二个文件夹（腐烂葡萄）
        for root, dirs, files in os.walk(folder2):
            for file in files:
                if file.endswith('.csv'):
                    file_path = os.path.join(root, file)
                    result = extract_force_data_and_calculate_stats(file_path)
                    if result is not None:
                        mean_force, std_force, rising_rate, peak_force, fft_data = result
                        features = prepare_features(
                            mean_force, std_force, rising_rate, peak_force, fft_data, max_fft_length)
                        if features is not None:
                            X.append(features)
                            y.append(1)

        X = np.array(X)
        y = np.array(y)

        # 打乱数据顺序
        X, y = shuffle(X, y, random_state=RANDOM_SEED)

        correct_predictions = 0
        total_predictions = len(X)

        for i in range(total_predictions):
            prediction = predict_grape_type(X[i], model, weights)
            if prediction is not None and prediction == y[i]:
                correct_predictions += 1

        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        print(f"模型在新数据集上的准确率: {accuracy * 100:.2f}%")
    