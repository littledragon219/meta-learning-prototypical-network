# data_processing.py
# 这个代码主要增加了数据增强，然后对截取的6～8.1秒信号进行峭度分析，找出冲击信号的时间。
# 注意：成熟葡萄信号能找到冲击时间点，而腐烂葡萄信号冲击不明显时，
# 可以尝试调整峭度阈值的乘数（threshold_factor）以便捕捉微弱冲击信号
# 对于sberry，grape的数据已经成功找出冲击时间点

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import kurtosis
from scipy.signal import find_peaks
from sklearn.preprocessing import StandardScaler

# 参数设置
SAMPLING_FREQUENCY = 500  # 单位 Hz
START_INDEX = int(6 * SAMPLING_FREQUENCY)
END_INDEX = int(8.1 * SAMPLING_FREQUENCY)
EXPECTED_LENGTH = END_INDEX - START_INDEX


def extract_features_with_fpeak(signal):
    """
    从信号中提取手工特征，并计算 Fpeak 特征。
    返回的特征包括：
      - initial_slope：信号初始斜率；
      - mean_val：信号均值；
      - variance：信号方差；
      - norm_energy：能量归一化值；
      - fpeak：峰值密度（即信号中峰的个数/信号长度）。
    """
    if len(signal) < 2:
        initial_slope = 0.0
    else:
        initial_slope = signal[1] - signal[0]
    mean_val = np.mean(signal)
    variance = np.var(signal)
    norm_energy = np.sum(signal ** 2) / len(signal)
    peaks, _ = find_peaks(signal)
    fpeak = len(peaks) / len(signal)
    return np.array([initial_slope, mean_val, variance, norm_energy, fpeak])


def augment_signal(signal, n_augments=2):
    """
    对整个输入信号进行数据增强：
      - 随机缩放（0.8～1.2之间的系数）；
      - 轻微时间平移（-10 到 10 个采样点），采用平移后边界处值填充。
    返回 n_augments 个增强信号。
    """
    augmented_signals = []
    for _ in range(n_augments):
        scale_factor = np.random.uniform(0.8, 1.2)
        scaled_signal = signal * scale_factor
        shift = np.random.randint(-10, 10)
        shifted_signal = np.roll(scaled_signal, shift)
        if shift > 0:
            shifted_signal[:shift] = shifted_signal[shift]
        elif shift < 0:
            shifted_signal[shift:] = shifted_signal[shift]
        augmented_signals.append(shifted_signal)
    return augmented_signals


def load_data(file_path):
    """
    从 CSV 文件中加载数据，自动选择包含 'Time' 和 'Force' 的列，
    并转换为浮点型 NumPy 数组。
    """
    data = pd.read_csv(file_path)
    time_col = None
    force_col = None
    for col in data.columns:
        if "Time" in col:
            time_col = col
        if "Force" in col:
            force_col = col
    if time_col is None or force_col is None:
        raise ValueError("未找到包含 'Time' 或 'Force' 的列，请检查文件格式。")
    time_data = np.array(data[time_col].values, dtype=float)
    force_data = np.array(data[force_col].values, dtype=float)
    return time_data, force_data


def kurtosis_analysis(time, signal, window_size=28):
    """
    采用滑动窗口计算信号的峭度，返回对应时间数组和峭度数组。
    """
    kurtosis_values = []
    for i in range(len(signal) - window_size + 1):
        window = signal[i:i + window_size]
        kurt = kurtosis(window)
        kurtosis_values.append(kurt)
    time_kurt = time[window_size - 1:]
    return time_kurt, np.array(kurtosis_values)


def get_shock_time_range(file_path, threshold_factor=1.0, window_size=5):
    """
    对单个文件在6~8.1秒区间进行峭度分析，确定冲击信号的时间范围。
    返回 start_time, end_time。
    """
    time, force = load_data(file_path)
    mask = (time >= 6) & (time <= 8.1)
    time_segment = time[mask]
    signal_segment = force[mask]
    if len(time_segment) == 0:
        print(f"{file_path}：指定时间区间内没有数据！")
        return None, None
    time_kurt, kurt_vals = kurtosis_analysis(time_segment, signal_segment, window_size=28)
    threshold = np.mean(kurt_vals) + threshold_factor * np.std(kurt_vals)
    shock_indices = np.where(kurt_vals > threshold)[0]
    if len(shock_indices) > 0:
        shock_times = time_kurt[shock_indices]
        first_shock_time = shock_times[0]
        last_shock_time = shock_times[-1]
        start_time = max(6, first_shock_time - 0.5)
        end_time = min(8.1, last_shock_time + 0.5)
        return start_time, end_time
    else:
        print(f"{file_path}：未检测到明显冲击。")
        return None, None


def extract_features(folder_path, label, domain_label, n_augments=2):
    """
    遍历文件夹下所有 CSV 文件，依据 get_shock_time_range 截取数据，
    对每个信号进行数据增强，并利用 extract_features_with_fpeak 提取特征。
    label：成熟度标签（如 ripe=0, rotten=1），domain_label：水果类型（如 grape=0, strawberry=1）。
    """
    handcrafted_features = []
    labels = []
    domain_labels = []
    signals = []
    files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    for f in files:
        file_path = os.path.join(folder_path, f)
        start_time, end_time = get_shock_time_range(file_path)
        if start_time is None or end_time is None:
            continue
        time, force = load_data(file_path)
        mask = (time >= start_time) & (time <= end_time)
        time_segment = time[mask]
        signal_segment = force[mask]
        if len(time_segment) == 0:
            print(f"{file_path}：指定时间区间内没有数据！")
            continue
        if len(signal_segment) < EXPECTED_LENGTH:
            original_indices = np.linspace(0, 1, len(signal_segment))
            new_indices = np.linspace(0, 1, EXPECTED_LENGTH)
            signal_segment = np.interp(new_indices, original_indices, signal_segment)
        elif len(signal_segment) > EXPECTED_LENGTH:
            signal_segment = signal_segment[:EXPECTED_LENGTH]
        augmented_signals = augment_signal(signal_segment, n_augments=n_augments)
        for aug_signal in augmented_signals:
            if len(aug_signal) != EXPECTED_LENGTH:
                original_indices = np.linspace(0, 1, len(aug_signal))
                new_indices = np.linspace(0, 1, EXPECTED_LENGTH)
                aug_signal = np.interp(new_indices, original_indices, aug_signal)
            features = extract_features_with_fpeak(aug_signal)
            handcrafted_features.append(features)
            labels.append(label)
            domain_labels.append(domain_label)
            signals.append(aug_signal)
    handcrafted_features = np.array(handcrafted_features)
    labels = np.array(labels)
    domain_labels = np.array(domain_labels)
    signals = np.array(signals)
    return handcrafted_features, labels, domain_labels, signals


def collect_fruit_data(fruit_name, ripe_folder, rotten_folder, domain_label):
    """
    分别采集成熟（label 0）与腐烂（label 1）的 CSV 信号数据，
    返回手工特征、成熟度标签、域标签和信号矩阵。
    """
    ripe_handcrafted, ripe_labels, ripe_domain_labels, ripe_signals = extract_features(ripe_folder, 0, domain_label,
                                                                                       n_augments=2)
    rotten_handcrafted, rotten_labels, rotten_domain_labels, rotten_signals = extract_features(rotten_folder, 1,
                                                                                               domain_label,
                                                                                               n_augments=2)
    handcrafted = np.vstack((ripe_handcrafted, rotten_handcrafted))
    labels = np.hstack((ripe_labels, rotten_labels))
    domain_labels = np.hstack((ripe_domain_labels, rotten_domain_labels))
    signals = np.vstack((ripe_signals, rotten_signals))
    return handcrafted, labels, domain_labels, signals


def prepare_dataset(fruit_data_list):
    """
    合并多个水果数据，并保证信号长度一致，同时使用 StandardScaler 对手工特征归一化。
    """
    all_handcrafted = []
    all_labels = []
    all_domain_labels = []
    all_signals = []
    expected_signal_length = fruit_data_list[0][3].shape[1]
    for handcrafted, labels, domain_labels, signals in fruit_data_list:
        if signals.shape[1] != expected_signal_length:
            signals = signals[:, :expected_signal_length] if signals.shape[1] > expected_signal_length else np.pad(
                signals, ((0, 0), (0, expected_signal_length - signals.shape[1])), 'constant')
        all_handcrafted.append(handcrafted)
        all_labels.append(labels)
        all_domain_labels.append(domain_labels)
        all_signals.append(signals)
    X_handcrafted = np.vstack(all_handcrafted)
    y = np.hstack(all_labels)
    domain_labels = np.hstack(all_domain_labels)
    signals = np.vstack(all_signals)
    scaler = StandardScaler()
    X_handcrafted = scaler.fit_transform(X_handcrafted)
    return X_handcrafted, y, domain_labels, signals


if __name__ == '__main__':
    folder_path = 'D:/CDIO/ripegrape'
    print("演示数据增强（对整个信号）...")
    sample_file = os.path.join(folder_path, '20250412_015043_sberry.csv')
    # 此处可调用数据增强和峭度分析函数进行验证
    print("\n演示文件夹中所有文件的峭度分析（6-8.1秒信号）：")
    # 可调用相关函数展示效果
    