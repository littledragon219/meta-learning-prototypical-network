# data_processing.py
# 这个代码主要增加了数据增强，然后对截取的6～8.1秒信号进行峭度分析，找出冲击信号的时间。
# 注意：成熟葡萄信号能找到冲击时间点，而腐烂葡萄信号冲击不明显时，
# 可以尝试调整峭度阈值的乘数（threshold_factor）以便捕捉微弱冲击信号
# 对于sberry，grape的数据已经成功找出冲击时间点

# datapreprocessing.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import kurtosis
from scipy.signal import find_peaks
import torch

# Default parameters (can be overridden)
DEFAULT_SAMPLING_FREQUENCY = 500  # Hz
DEFAULT_SIGNAL_START_TIME = 6.0   # seconds
DEFAULT_SIGNAL_END_TIME = 8.1     # seconds
DEFAULT_KURTOSIS_WINDOW = 28
DEFAULT_KURTOSIS_THRESHOLD_FACTOR = 1.0
DEFAULT_AUGMENTATIONS = 2 # Number of augmentations per original signal

def calculate_expected_length(start_time, end_time, sampling_freq):
    """Calculates the expected signal length based on time and frequency."""
    return int((end_time - start_time) * sampling_freq)

def extract_handcrafted_features(signal):
    """Extracts 5 handcrafted features from a signal segment."""
    if len(signal) < 2:
        return np.zeros(5) # Return zeros if signal is too short
    initial_slope = signal[1] - signal[0]
    mean_val = np.mean(signal)
    variance = np.var(signal)
    norm_energy = np.sum(signal ** 2) / len(signal) if len(signal) > 0 else 0
    peaks, _ = find_peaks(signal)
    fpeak = len(peaks) / len(signal) if len(signal) > 0 else 0
    return np.array([initial_slope, mean_val, variance, norm_energy, fpeak])

def augment_signal(signal, n_augments=DEFAULT_AUGMENTATIONS):
    """Applies random scaling and shifting for data augmentation."""
    augmented_signals = []
    if len(signal) == 0: # Handle empty signals
        return [signal] * n_augments # Return copies of empty array

    for _ in range(n_augments):
        scale_factor = np.random.uniform(0.8, 1.2)
        scaled_signal = signal * scale_factor
        shift = np.random.randint(-10, 11) # Inclusive range -10 to 10
        shifted_signal = np.roll(scaled_signal, shift)
        if shift > 0:
            shifted_signal[:shift] = shifted_signal[shift]
        elif shift < 0:
            shifted_signal[shift:] = shifted_signal[shift-1] if shift < -1 else shifted_signal[-1] # Avoid index error

        augmented_signals.append(shifted_signal)
    return augmented_signals

def load_csv_data(file_path):
    """Loads time and force data from a CSV file."""
    try:
        data = pd.read_csv(file_path)
        time_col = next((col for col in data.columns if "Time" in col), None)
        force_col = next((col for col in data.columns if "Force" in col), None)
        if time_col is None or force_col is None:
            print(f"Warning: Could not find Time or Force columns in {file_path}. Skipping.")
            return None, None
        time_data = np.array(data[time_col].values, dtype=float)
        force_data = np.array(data[force_col].values, dtype=float)
        return time_data, force_data
    except Exception as e:
        print(f"Error loading {file_path}: {e}. Skipping.")
        return None, None

def kurtosis_analysis(time, signal, window_size=DEFAULT_KURTOSIS_WINDOW):
    """Calculates sliding window kurtosis."""
    if len(signal) < window_size:
        return np.array([]), np.array([])
    kurtosis_values = []
    # Correctly calculate indices for time_kurt
    half_window = window_size // 2
    time_indices = np.arange(half_window, len(signal) - half_window + (window_size % 2)) # Center window time

    for i in range(len(signal) - window_size + 1):
        window = signal[i:i + window_size]
        if np.std(window) == 0: # Avoid division by zero if window is flat
            kurt = 0
        else:
             kurt = kurtosis(window, fisher=False) # Use Pearson's kurtosis (normal=3)
        kurtosis_values.append(kurt)

    # Ensure time_kurt aligns with kurtosis_values (adjust index for time)
    valid_time_indices = np.arange(window_size - 1, len(signal))
    if len(valid_time_indices) != len(kurtosis_values):
         # Fallback or simpler alignment if centering logic is complex/buggy
         time_kurt = time[window_size - 1 : window_size -1 + len(kurtosis_values)]
    else:
        time_kurt = time[valid_time_indices]

    return time_kurt, np.array(kurtosis_values)


def get_shock_time_range(time, force,
                         start_scan=DEFAULT_SIGNAL_START_TIME,
                         end_scan=DEFAULT_SIGNAL_END_TIME,
                         freq=DEFAULT_SAMPLING_FREQUENCY,
                         kurt_window=DEFAULT_KURTOSIS_WINDOW,
                         kurt_factor=DEFAULT_KURTOSIS_THRESHOLD_FACTOR,
                         min_duration=0.1): # Minimum duration of interest
    """Analyzes kurtosis in a segment to find the shock time range."""
    mask = (time >= start_scan) & (time <= end_scan)
    time_segment = time[mask]
    signal_segment = force[mask]

    if len(time_segment) < kurt_window:
        # print(f"Warning: Not enough data points ({len(time_segment)}) in scan range [{start_scan}s, {end_scan}s] for kurtosis window {kurt_window}. Using full range.")
        return start_scan, end_scan # Return original range if too short

    time_kurt, kurt_vals = kurtosis_analysis(time_segment, signal_segment, window_size=kurt_window)

    if len(kurt_vals) == 0:
        # print(f"Warning: Kurtosis analysis returned no values for scan range [{start_scan}s, {end_scan}s]. Using full range.")
        return start_scan, end_scan

    # Use a threshold relative to the median or mean kurtosis
    threshold = np.median(kurt_vals) + kurt_factor * np.std(kurt_vals)
    # Ensure threshold is reasonably above baseline (e.g., Pearson's normal = 3)
    threshold = max(threshold, 3.0 + 1.0 * np.std(kurt_vals))

    shock_indices = np.where(kurt_vals > threshold)[0]

    if len(shock_indices) > 0:
        # Find corresponding times
        shock_times = time_kurt[shock_indices]
        first_shock_time_kurt = shock_times[0]
        last_shock_time_kurt = shock_times[-1]

        # Map kurtosis time back to original signal time indices more carefully
        # Find the index in the original time_segment that corresponds to the start/end kurtosis times
        start_idx_in_segment = np.searchsorted(time_segment, first_shock_time_kurt) - kurt_window // 2
        end_idx_in_segment = np.searchsorted(time_segment, last_shock_time_kurt) + kurt_window // 2
        start_idx_in_segment = max(0, start_idx_in_segment) # Clamp to bounds
        end_idx_in_segment = min(len(time_segment)-1, end_idx_in_segment)

        start_time = time_segment[start_idx_in_segment]
        end_time = time_segment[end_idx_in_segment]

        # Expand slightly and ensure minimum duration
        duration = end_time - start_time
        expansion = max(0, (min_duration - duration) / 2.0) # Expand to meet min_duration

        final_start_time = max(start_scan, start_time - expansion)
        final_end_time = min(end_scan, end_time + expansion)

        # Ensure we don't exceed original scan boundaries
        final_start_time = max(start_scan, final_start_time)
        final_end_time = min(end_scan, final_end_time)

        # Final check: Ensure end > start
        if final_end_time <= final_start_time:
             final_start_time = start_scan
             final_end_time = end_scan

        # print(f"Shock detected: Kurtosis range [{first_shock_time_kurt:.2f}s, {last_shock_time_kurt:.2f}s] -> Signal range [{final_start_time:.2f}s, {final_end_time:.2f}s]")
        return final_start_time, final_end_time
    else:
        # print(f"No significant shock detected via kurtosis in scan range [{start_scan}s, {end_scan}s]. Using full range.")
        return start_scan, end_scan # Return original range if no shock

def process_file(file_path, maturity_label, domain_label, expected_length,
                 start_scan=DEFAULT_SIGNAL_START_TIME,
                 end_scan=DEFAULT_SIGNAL_END_TIME,
                 freq=DEFAULT_SAMPLING_FREQUENCY,
                 n_augments=DEFAULT_AUGMENTATIONS):
    """Loads, finds shock, segments, augments, and extracts features from one file."""
    time, force = load_csv_data(file_path)
    if time is None:
        return [], [], [], [] # Return empty lists if loading failed

    # Find the relevant time window using kurtosis
    start_time, end_time = get_shock_time_range(time, force, start_scan, end_scan, freq)

    # Extract the signal segment
    mask = (time >= start_time) & (time <= end_time)
    signal_segment = force[mask]

    if len(signal_segment) == 0:
        print(f"Warning: No data in determined range [{start_time:.2f}s, {end_time:.2f}s] for {file_path}. Skipping.")
        return [], [], [], []

    # --- Handle signal length difference ---
    if len(signal_segment) != expected_length:
        if len(signal_segment) < expected_length:
            # Pad: Using reflection padding might be better than zeros
            pad_width = expected_length - len(signal_segment)
            signal_segment = np.pad(signal_segment, (0, pad_width), 'reflect')
        else:
            # Truncate (from center or edges? Let's truncate from end for simplicity)
            signal_segment = signal_segment[:expected_length]

    # Create original sample + augmentations
    all_signals_for_file = [signal_segment] + augment_signal(signal_segment, n_augments=n_augments)

    handcrafted = []
    processed_signals = []
    maturity_labels = []
    domain_labels = []

    for sig in all_signals_for_file:
         # Ensure augmented signals also have the correct length (should be okay if input was correct)
        if len(sig) != expected_length:
             # This shouldn't happen often if padding/truncation was done correctly before augmentation
             if len(sig) < expected_length:
                 pad_width = expected_length - len(sig)
                 sig = np.pad(sig, (0, pad_width), 'reflect')
             else:
                 sig = sig[:expected_length]

        features = extract_handcrafted_features(sig)
        handcrafted.append(features)
        processed_signals.append(sig)
        maturity_labels.append(maturity_label) # 0 for ripe, 1 for rotten
        domain_labels.append(domain_label)

    return handcrafted, processed_signals, maturity_labels, domain_labels


def collect_data_from_folders(fruit_name, ripe_folder, rotten_folder, domain_label,
                               expected_length, n_augments=DEFAULT_AUGMENTATIONS):
    """Collects and processes data from ripe and rotten folders for a fruit."""
    all_handcrafted = []
    all_signals = []
    all_maturity_labels = []
    all_domain_labels = []

    print(f"Processing {fruit_name} (Domain {domain_label})...")
    # Process Ripe (Label 0)
    print(f"  Processing Ripe folder: {ripe_folder}")
    ripe_files = [f for f in os.listdir(ripe_folder) if f.endswith('.csv')]
    for f in ripe_files:
        file_path = os.path.join(ripe_folder, f)
        h, s, m, d = process_file(file_path, 0, domain_label, expected_length, n_augments=n_augments)
        all_handcrafted.extend(h)
        all_signals.extend(s)
        all_maturity_labels.extend(m)
        all_domain_labels.extend(d)
    print(f"    Found {len(ripe_files)} ripe files -> {len(all_handcrafted)} samples (incl. augmentations).")

    # Process Rotten (Label 1)
    rotten_start_idx = len(all_handcrafted) # Keep track for stats
    print(f"  Processing Rotten folder: {rotten_folder}")
    rotten_files = [f for f in os.listdir(rotten_folder) if f.endswith('.csv')]
    for f in rotten_files:
        file_path = os.path.join(rotten_folder, f)
        h, s, m, d = process_file(file_path, 1, domain_label, expected_length, n_augments=n_augments)
        all_handcrafted.extend(h)
        all_signals.extend(s)
        all_maturity_labels.extend(m)
        all_domain_labels.extend(d)
    rotten_count = len(all_handcrafted) - rotten_start_idx
    print(f"    Found {len(rotten_files)} rotten files -> {rotten_count} samples (incl. augmentations).")

    if not all_handcrafted: # Check if any data was processed
        print(f"Warning: No data collected for {fruit_name}. Check paths and file contents.")
        return np.array([]), np.array([]), np.array([]), np.array([])

    return (np.array(all_handcrafted), np.array(all_signals),
            np.array(all_maturity_labels), np.array(all_domain_labels))


if __name__ == '__main__':
    # Example Usage (demonstration)
    print("Demonstrating data collection...")
    EXPECTED_LENGTH = calculate_expected_length(
        DEFAULT_SIGNAL_START_TIME,
        DEFAULT_SIGNAL_END_TIME,
        DEFAULT_SAMPLING_FREQUENCY
    )

    # --- Define dummy paths for demonstration ---
    # Create dummy folders and files if they don't exist
    dummy_base = "dummy_data"
    os.makedirs(os.path.join(dummy_base, "grape_ripe"), exist_ok=True)
    os.makedirs(os.path.join(dummy_base, "grape_rotten"), exist_ok=True)
    # Create dummy CSV files
    time_dummy = np.linspace(0, 10, 10 * DEFAULT_SAMPLING_FREQUENCY)
    force_dummy_ripe = np.sin(time_dummy * 5) + np.random.randn(len(time_dummy)) * 0.1
    force_dummy_ripe[int(7*DEFAULT_SAMPLING_FREQUENCY):int(7.2*DEFAULT_SAMPLING_FREQUENCY)] += 5 # Add a 'shock'
    force_dummy_rotten = np.sin(time_dummy * 2) + np.random.randn(len(time_dummy)) * 0.2
    pd.DataFrame({'Time': time_dummy, 'Force': force_dummy_ripe}).to_csv(os.path.join(dummy_base, "grape_ripe", "ripe_sample1.csv"), index=False)
    pd.DataFrame({'Time': time_dummy, 'Force': force_dummy_rotten}).to_csv(os.path.join(dummy_base, "grape_rotten", "rotten_sample1.csv"), index=False)

    # --- Set paths to the dummy data ---
    grape_ripe_path = os.path.join(dummy_base, "grape_ripe")
    grape_rotten_path = os.path.join(dummy_base, "grape_rotten")

    h_feat, signals, mat_labels, dom_labels = collect_data_from_folders(
        fruit_name="grape",
        ripe_folder=grape_ripe_path,
        rotten_folder=grape_rotten_path,
        domain_label=0, # Grape = domain 0
        expected_length=EXPECTED_LENGTH,
        n_augments=2 # 1 original + 2 augmented = 3 samples per file
    )

    if h_feat.size > 0:
        print("\nCollected Data Shapes:")
        print(f"  Handcrafted Features: {h_feat.shape}")      # (num_samples, num_handcrafted_features)
        print(f"  Signals:            {signals.shape}")       # (num_samples, expected_length)
        print(f"  Maturity Labels:    {mat_labels.shape}")    # (num_samples,) -> Binary (0 or 1)
        print(f"  Domain Labels:      {dom_labels.shape}")     # (num_samples,) -> Domain index (0, 1, 2...)

        print("\nSample Data:")
        print("  Handcrafted Features (first sample):\n", h_feat[0])
        print("  Maturity Label (first sample):", mat_labels[0])
        print("  Domain Label (first sample):", dom_labels[0])

        # Plot the first signal segment
        plt.figure(figsize=(10, 4))
        plt.plot(signals[0])
        plt.title(f"Example Processed Signal Segment (Length: {signals.shape[1]})")
        plt.xlabel("Sample Index")
        plt.ylabel("Force (Processed)")
        plt.grid(True)
        plt.show()
    else:
        print("No data collected in demonstration.")
    # Clean up dummy data
    # import shutil
    # shutil.rmtree(dummy_base)