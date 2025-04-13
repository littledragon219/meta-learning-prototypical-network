# test.py
import os
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report
import joblib # To load the scaler
import matplotlib.pyplot as plt
from scipy.stats import kurtosis
from scipy.signal import find_peaks

# Import necessary components from your project files
from model import FusionDomainAdaptationNetwork # Needs model.py
from utils import calculate_metrics, plot_confusion_matrix, visualize_embeddings, robust_imputation # Needs utils.py
from datapreprocessing import EXPECTED_LENGTH
# We might need some functions from datapreprocessing, let's copy/adapt them here
# or import them if datapreprocessing.py is stable and accessible

# --- Configuration ---
# Paths for the saved model, scaler, and new test data
MODEL_LOAD_PATH = 'final_model_and_config_v2.pth'
SCALER_LOAD_PATH = 'scaler_v2.joblib'

# Define where your *new* test data is located.
# Structure: Dict where keys are fruit names, values are dicts mapping maturity state to folder path.
# Example:
NEW_TEST_DATA_PATHS = {
    "grape": {
        "ripe_test": r"D:/大二下学期/CDIO/grape_ripe_test",   # CHANGE THIS PATH
        "rotten_test": r"D:/大二下学期/CDIO/grape_rotten_test" # CHANGE THIS PATH
    },
    "strawberry": {
        "ripe_test": r"D:/大二下学期/CDIO/strawberry_ripe_test", # CHANGE THIS PATH
        "rotten_test": r"D:/大二下学期/CDIO/strawberry_rotten_test" # CHANGE THIS PATH
    },
    "tomato": {
        "ripe_test": r"D:/大二下学期/CDIO/tomato_ripe_test", # CHANGE THIS PATH
        "rotten_test": r"D:/大二下学期/CDIO/tomato_rotten_test"
    },
    # Add more fruits or maturity levels if needed
}

# --- Data Loading and Preprocessing Functions (Adapted from datapreprocessing.py) ---
# Parameters needed from training (should match those used in datapreprocessing.py)
SAMPLING_FREQUENCY = 250
START_INDEX = int(6 * SAMPLING_FREQUENCY)
END_INDEX = int(8.1 * SAMPLING_FREQUENCY)
EXPECTED_LENGTH = END_INDEX - START_INDEX # Required signal length for CNN

def load_data_test(file_path):
    """ Loads time and force data from CSV. """
    try:
        data = pd.read_csv(file_path)
        time_col = None
        force_col = None
        for col in data.columns:
            if "Time" in col: time_col = col
            if "Force" in col: force_col = col
        if time_col is None or force_col is None:
            print(f"警告: 文件 {os.path.basename(file_path)} 中未找到 Time 或 Force 列。跳过。")
            return None, None
        time_data = np.array(data[time_col].values, dtype=float)
        force_data = np.array(data[force_col].values, dtype=float)
        return time_data, force_data
    except Exception as e:
        print(f"错误: 读取文件 {file_path} 时出错: {e}")
        return None, None

def kurtosis_analysis_test(time, signal, window_size=28):
    """ Calculates kurtosis using a sliding window. """
    if len(signal) < window_size: return np.array([]), np.array([])
    kurtosis_values = [kurtosis(signal[i:i + window_size]) for i in range(len(signal) - window_size + 1)]
    time_kurt = time[window_size - 1:]
    return time_kurt, np.array(kurtosis_values)

def get_shock_time_range_test(time, force, file_path_for_log, threshold_factor=1.0, window_size=5):
    """ Analyzes kurtosis in the 6-8.1s segment to find shock time range. """
    mask = (time >= 6) & (time <= 8.1)
    time_segment = time[mask]
    signal_segment = force[mask]
    if len(time_segment) == 0:
        # print(f"文件 {os.path.basename(file_path_for_log)}: 6-8.1s 区间无数据。")
        return None, None
    time_kurt, kurt_vals = kurtosis_analysis_test(time_segment, signal_segment, window_size=28)
    if len(kurt_vals) == 0:
        # print(f"文件 {os.path.basename(file_path_for_log)}: 无法计算峭度。")
        return None, None

    # Handle potential NaN/inf in kurtosis values before calculating threshold
    kurt_vals_clean = kurt_vals[np.isfinite(kurt_vals)]
    if len(kurt_vals_clean) < 2: # Need at least 2 points for std dev
        # print(f"文件 {os.path.basename(file_path_for_log)}: 清理后的峭度值过少。")
        return None, None # Or return the default full range [6, 8.1]?

    threshold = np.mean(kurt_vals_clean) + threshold_factor * np.std(kurt_vals_clean)
    shock_indices = np.where(kurt_vals > threshold)[0] # Use original kurt_vals for indices

    if len(shock_indices) > 0:
        shock_times = time_kurt[shock_indices]
        first_shock_time = shock_times[0]
        last_shock_time = shock_times[-1]
        start_time = max(6, first_shock_time - 0.5)
        end_time = min(8.1, last_shock_time + 0.5)
        return start_time, end_time
    else:
        # print(f"文件 {os.path.basename(file_path_for_log)}: 未检测到明显冲击。")
        # Fallback: maybe return the default range? Or None? Returning None for now.
        return None, None

def extract_features_with_fpeak_test(signal):
    """ Extracts handcrafted features including Fpeak. """
    if len(signal) < 2: initial_slope = 0.0
    else: initial_slope = signal[1] - signal[0]
    mean_val = np.mean(signal)
    variance = np.var(signal)
    norm_energy = np.sum(signal ** 2) / max(1, len(signal)) # Avoid division by zero
    peaks, _ = find_peaks(signal)
    fpeak = len(peaks) / max(1, len(signal)) # Avoid division by zero
    return np.array([initial_slope, mean_val, variance, norm_energy, fpeak])

def process_test_folder(folder_path, label, domain_label, expected_signal_length):
    """ Processes all CSV files in a folder for testing. No augmentation. """
    handcrafted_features = []
    labels = []
    domain_labels_list = []
    signals = []
    print(f"  Processing folder: {folder_path} (Label: {label}, Domain: {domain_label})")
    if not os.path.isdir(folder_path):
        print(f"  警告: 路径不存在或不是文件夹: {folder_path}")
        return np.array([]), np.array([]), np.array([]), np.array([])

    files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    if not files:
        print(f"  警告: 文件夹中未找到 .csv 文件: {folder_path}")

    for f in files:
        file_path = os.path.join(folder_path, f)
        time, force = load_data_test(file_path)
        if time is None: continue

        start_time, end_time = get_shock_time_range_test(time, force, file_path)
        if start_time is None or end_time is None:
             # Fallback: Use the default 6-8.1s range if shock detection fails
             start_time, end_time = 6.0, 8.1
             # print(f"    使用默认时间范围 [6.0, 8.1] 对于文件: {f}")


        mask = (time >= start_time) & (time <= end_time)
        signal_segment = force[mask]

        if len(signal_segment) == 0:
            # print(f"    警告: 文件 {f} 在时间段 [{start_time:.2f}, {end_time:.2f}] 内无数据。跳过。")
            continue

        # Adjust signal length to match training
        if len(signal_segment) != expected_signal_length:
            if len(signal_segment) > expected_signal_length:
                signal_segment = signal_segment[:expected_signal_length]
            else:
                signal_segment = np.pad(signal_segment, (0, expected_signal_length - len(signal_segment)), 'constant', constant_values=0)

        # Extract features
        features = extract_features_with_fpeak_test(signal_segment)

        handcrafted_features.append(features)
        labels.append(label)
        domain_labels_list.append(domain_label)
        signals.append(signal_segment)

    if not handcrafted_features: # If no files were successfully processed
         return np.array([]), np.array([]), np.array([]), np.array([])

    return np.array(handcrafted_features), np.array(labels), np.array(domain_labels_list), np.array(signals)

# --- Main Testing Logic ---
if __name__ == '__main__':
    print("--- 开始测试流程 (V2 Model) ---")

    # --- Load Model and Scaler (Points to new files) ---
    print("加载模型配置和状态...")
    if not os.path.exists(MODEL_LOAD_PATH):
        raise FileNotFoundError(f"未找到模型文件: {MODEL_LOAD_PATH}")
    # Load from CPU first in case the saving device differs
    saved_data = torch.load(MODEL_LOAD_PATH, map_location=torch.device('cpu'))
    model_config = saved_data['config']
    model_state_dict = saved_data['state_dict']

    print("加载 Scaler...")
    if not os.path.exists(SCALER_LOAD_PATH):
        raise FileNotFoundError(f"未找到 Scaler 文件: {SCALER_LOAD_PATH}")
    scaler = joblib.load(SCALER_LOAD_PATH)

    # Extract info (unchanged)
    cnn_input_len_test = model_config['cnn_input_length']
    num_maturity_classes_test = model_config['num_classes']
    domain_classes_count_test = model_config['domain_classes']
    print(f"模型配置加载成功: Signal Length={cnn_input_len_test}, Maturity Classes={num_maturity_classes_test}, Domain Classes={domain_classes_count_test}")

    # --- Load New Test Data (Unchanged logic) ---
    # ... (rest of the data loading, scaling, tensor prep) ...
    print("\n加载新的测试数据...")
    all_test_handcrafted = []
    all_test_labels = []
    all_test_domain_labels = []
    all_test_signals = []
    test_fruit_names = list(NEW_TEST_DATA_PATHS.keys())
    test_domain_mapping = {name: i for i, name in enumerate(test_fruit_names)}
    maturity_map = {}
    if len(test_domain_mapping) != domain_classes_count_test:
         print(f"警告: 测试数据中的领域数量 ({len(test_domain_mapping)}) 与训练模型时的领域数量 ({domain_classes_count_test}) 不匹配。")

    for fruit_name, fruit_paths in NEW_TEST_DATA_PATHS.items():
        if fruit_name not in test_domain_mapping: continue
        domain_id = test_domain_mapping[fruit_name]
        for maturity_key, folder_path in fruit_paths.items():
             combined_key = f"{fruit_name}_{maturity_key}"
             if combined_key not in maturity_map:
                  is_rotten = "rotten" in maturity_key.lower()
                  label_id = domain_id * 2 + (1 if is_rotten else 0)
                  maturity_map[combined_key] = label_id
                  print(f"  映射: {combined_key} -> Label ID: {label_id}")
             current_label = maturity_map[combined_key]
             h_feat, lab, dom, sig = process_test_folder(folder_path, current_label, domain_id, EXPECTED_LENGTH)
             if h_feat.shape[0] > 0:
                 all_test_handcrafted.append(h_feat); all_test_labels.append(lab); all_test_domain_labels.append(dom); all_test_signals.append(sig)

    if not all_test_handcrafted: raise ValueError("未能从任何指定的 NEW_TEST_DATA_PATHS 加载有效数据。")
    X_handcrafted_test_new = np.vstack(all_test_handcrafted); y_label_test_new = np.hstack(all_test_labels); y_domain_test_new = np.hstack(all_test_domain_labels); signals_test_new = np.vstack(all_test_signals)
    print(f"\n总共加载了 {X_handcrafted_test_new.shape[0]} 个新的测试样本。")
    print("应用加载的 Scaler 到新的手工特征...")
    X_handcrafted_test_new_scaled = scaler.transform(X_handcrafted_test_new)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    X_hand_test_new_t = torch.tensor(X_handcrafted_test_new_scaled, dtype=torch.float32).to(device)
    signals_test_new_t = torch.tensor(signals_test_new, dtype=torch.float32).unsqueeze(1).to(device)
    y_label_test_new_t = torch.tensor(y_label_test_new, dtype=torch.long).to(device)
    y_domain_test_new_t = torch.tensor(y_domain_test_new, dtype=torch.long).to(device)


    # --- Instantiate Model using loaded config ---
    print("实例化模型并加载状态...")
    model_test = FusionDomainAdaptationNetwork(**model_config).to(device) # Instantiate with loaded config
    model_test.load_state_dict(model_state_dict)
    model_test.eval()

    # --- Inference, Evaluation, Visualization (Unchanged logic) ---
    # ... (rest of the script: inference, evaluation, visualization) ...
    print("在新测试集上执行推理...")
    with torch.no_grad():
        class_logits_new, domain_logits_new, fused_feats_new = model_test(
            X_hand_test_new_t, signals_test_new_t,
            domain_labels=y_domain_test_new_t, lambda_grl=0.0
        )
        y_pred_label_new = torch.argmax(class_logits_new, dim=1).cpu().numpy()
        y_pred_domain_new = None
        if domain_logits_new is not None:
            y_pred_domain_new = torch.argmax(domain_logits_new, dim=1).cpu().numpy()

    y_true_label_new_np = y_label_test_new_t.cpu().numpy()
    y_true_domain_new_np = y_domain_test_new_t.cpu().numpy()

    print("\n--- 新测试集评估结果 ---")
    test_label_names = [""] * num_maturity_classes_test
    num_maturity_states = 2
    for fruit_name, domain_id in test_domain_mapping.items():
        for maturity_key, folder_path in NEW_TEST_DATA_PATHS[fruit_name].items():
             combined_key = f"{fruit_name}_{maturity_key}"
             if combined_key in maturity_map:
                  label_id = maturity_map[combined_key]
                  maturity_desc = maturity_key.split('_')[0].capitalize()
                  fruit_desc = fruit_name.capitalize()
                  if label_id < len(test_label_names): test_label_names[label_id] = f"{maturity_desc} {fruit_desc}"
    for i, name in enumerate(test_label_names):
        if not name: test_label_names[i] = f"Class {i}"
    test_domain_names = [name.capitalize() for name in test_fruit_names]

    if y_pred_domain_new is not None:
        print(f"\n🍇 领域分类 ({'/'.join(test_domain_names)}):")
        print(classification_report(y_true_domain_new_np, y_pred_domain_new, target_names=test_domain_names, zero_division=0))
        plot_confusion_matrix(y_true_domain_new_np, y_pred_domain_new, classes=test_domain_names)

    print(f"\n🍓 类别分类 ({'/'.join(test_label_names[:4])}...):")
    print(classification_report(y_true_label_new_np, y_pred_label_new, target_names=test_label_names, zero_division=0))
    calculate_metrics(y_true_label_new_np, y_pred_label_new)
    plot_confusion_matrix(y_true_label_new_np, y_pred_label_new, classes=test_label_names)

    print("\n在新测试集上进行 t-SNE 可视化...")
    if fused_feats_new is not None and fused_feats_new.shape[0] > 0:
        prototypes_vis_test = torch.zeros(num_maturity_classes_test, fused_feats_new.shape[1]).to(device)
        visualize_embeddings(
            model=model_test, handcrafted_tensor=X_hand_test_new_t, signal_tensor=signals_test_new_t,
            y=y_true_label_new_np, prototypes=prototypes_vis_test, device=device,
            title_suffix="New Test Set Embeddings (V2 Model)"
        )
    else: print("无法进行 t-SNE 可视化，新测试集融合特征为空或不存在。")

    print("\n--- 测试流程结束 ---")