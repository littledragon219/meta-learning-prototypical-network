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

# Default parameters (can be overridden by config)
DEFAULT_SAMPLING_FREQUENCY = 250 # Hz - Will be overridden by config
DEFAULT_SIGNAL_START_TIME = 6.0   # seconds - Will be overridden by config
DEFAULT_SIGNAL_END_TIME = 8.1     # seconds - Will be overridden by config
DEFAULT_KURTOSIS_WINDOW = 28
DEFAULT_KURTOSIS_THRESHOLD_FACTOR = 1.0
DEFAULT_AUGMENTATIONS = 1 # Number of augmentations per original signal - Will be overridden by config

# --- OPTIMIZATION NOTE ---
# Kurtosis calculation can be somewhat slow on CPU for many files.
# If speed is critical and a fixed window (e.g., 6s-8.1s) is acceptable,
# you could simplify `get_shock_time_range` to always return `start_scan`, `end_scan`.
# However, the current method aims to find the most relevant signal part.

def calculate_expected_length(start_time, end_time, sampling_freq):
    """Calculates the expected signal length based on time and frequency."""
    # Ensure inputs are numeric
    try:
        start_time_f = float(start_time)
        end_time_f = float(end_time)
        sampling_freq_f = int(sampling_freq)
    except (ValueError, TypeError) as e:
        print(f"Error converting time/frequency to numbers: {e}. Using defaults.")
        # Provide some default fallback length or raise error
        return int((8.1 - 6.0) * 250) # Fallback based on typical values

    return int((end_time_f - start_time_f) * sampling_freq_f)

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
        # Return empty arrays with the same structure
        return [np.array([])] * n_augments

    for _ in range(n_augments):
        scale_factor = np.random.uniform(0.8, 1.2)
        scaled_signal = signal * scale_factor
        # Shift amount relative to signal length (e.g., max 5% shift)
        max_shift = max(1, int(len(signal) * 0.05))
        shift = np.random.randint(-max_shift, max_shift + 1)
        # shift = np.random.randint(-10, 11) # Original fixed shift

        shifted_signal = np.roll(scaled_signal, shift)
        if shift > 0:
            shifted_signal[:shift] = shifted_signal[shift] # Pad start
        elif shift < 0:
            shifted_signal[shift:] = shifted_signal[shift-1] if len(signal) + shift > 0 else shifted_signal[-1] # Pad end


        augmented_signals.append(shifted_signal)
    return augmented_signals


def load_csv_data(file_path):
    """Loads time and force data from a CSV file."""
    try:
        data = pd.read_csv(file_path)
        time_col = next((col for col in data.columns if "Time" in col or "TIME" in col), None) # Case insensitive search
        force_col = next((col for col in data.columns if "Force" in col or "FORCE" in col), None) # Case insensitive search
        if time_col is None or force_col is None:
            print(f"Warning: Could not find Time or Force columns in {file_path}. Skipping.")
            return None, None
        # Attempt to convert to numeric, coercing errors to NaN, then drop NaNs
        time_data_pd = pd.to_numeric(data[time_col], errors='coerce')
        force_data_pd = pd.to_numeric(data[force_col], errors='coerce')
        valid_indices = time_data_pd.notna() & force_data_pd.notna()
        time_data = time_data_pd[valid_indices].values.astype(float)
        force_data = force_data_pd[valid_indices].values.astype(float)

        if len(time_data) == 0:
            print(f"Warning: No valid numeric data found in {file_path} after cleaning. Skipping.")
            return None, None

        return time_data, force_data
    except Exception as e:
        print(f"Error loading {file_path}: {e}. Skipping.")
        return None, None

def kurtosis_analysis(time, signal, window_size=DEFAULT_KURTOSIS_WINDOW):
    """Calculates sliding window kurtosis."""
    if len(signal) < window_size:
        return np.array([]), np.array([]) # Return empty arrays if signal is too short
    kurtosis_values = []
    # Correctly calculate indices for time_kurt
    half_window = window_size // 2
    # Center window time - Ensure indices are within bounds
    start_idx = half_window
    end_idx = len(signal) - half_window + (window_size % 2 > 0) # Adjust for odd window size
    time_indices = np.arange(start_idx, end_idx)


    kurtosis_values = np.array([kurtosis(signal[i : i + window_size], fisher=False) if np.std(signal[i : i + window_size]) > 1e-6 else 3.0 for i in range(len(signal) - window_size + 1)])

    # Ensure time_kurt aligns with kurtosis_values
    if len(time_indices) != len(kurtosis_values):
        # Fallback if centering logic has issues or edge cases
        time_kurt = time[window_size - 1 : window_size -1 + len(kurtosis_values)]
    else:
         # Ensure the time indices are valid for the original time array
        time_kurt = time[time_indices[:len(kurtosis_values)]] # Slice time_indices to match kurt_vals length

    # Ensure time_kurt and kurtosis_values have the same length
    min_len = min(len(time_kurt), len(kurtosis_values))
    return time_kurt[:min_len], kurtosis_values[:min_len]

def get_shock_time_range(time, force,
                         start_scan=DEFAULT_SIGNAL_START_TIME,
                         end_scan=DEFAULT_SIGNAL_END_TIME,
                         freq=DEFAULT_SAMPLING_FREQUENCY,
                         kurt_window=DEFAULT_KURTOSIS_WINDOW,
                         kurt_factor=DEFAULT_KURTOSIS_THRESHOLD_FACTOR,
                         min_duration=0.1): # Minimum duration of interest
    """Analyzes kurtosis in a segment to find the shock time range."""
    if len(time) == 0: return start_scan, end_scan # Handle empty input

    mask = (time >= start_scan) & (time <= end_scan)
    time_segment = time[mask]
    signal_segment = force[mask]

    if len(time_segment) < kurt_window or len(signal_segment) < kurt_window:
        # print(f"Warning: Not enough data points ({len(time_segment)}) in scan range [{start_scan}s, {end_scan}s] for kurtosis window {kurt_window}. Using full range.")
        return start_scan, end_scan # Return original range if too short

    time_kurt, kurt_vals = kurtosis_analysis(time_segment, signal_segment, window_size=kurt_window)

    if len(kurt_vals) == 0:
        # print(f"Warning: Kurtosis analysis returned no values for scan range [{start_scan}s, {end_scan}s]. Using full range.")
        return start_scan, end_scan

    # Use a threshold relative to the median or mean kurtosis
    median_kurt = np.median(kurt_vals)
    std_kurt = np.std(kurt_vals)
    threshold = median_kurt + kurt_factor * std_kurt
    # Ensure threshold is reasonably above baseline (e.g., Pearson's normal = 3)
    # Use a slightly more robust threshold against outliers
    threshold = max(threshold, median_kurt + 1.0) # At least 1 above median
    threshold = max(threshold, 3.5) # Ensure it's above normal distribution kurtosis


    shock_indices = np.where(kurt_vals > threshold)[0]

    if len(shock_indices) > 0:
        # Find corresponding times in time_kurt
        shock_times_kurt = time_kurt[shock_indices]
        first_shock_time_kurt = shock_times_kurt[0]
        last_shock_time_kurt = shock_times_kurt[-1]

        # Map kurtosis time back to original signal time indices more carefully
        # Find the index in the original time_segment that corresponds to the start/end kurtosis times
        # time_segment corresponds to the kurt_vals time frame (time_kurt)
        start_idx_in_segment = np.searchsorted(time_segment, first_shock_time_kurt, side='left')
        end_idx_in_segment = np.searchsorted(time_segment, last_shock_time_kurt, side='right') # Use right for end time

        # Extend the window slightly around the detected shock times, relative to kurtosis window
        kurt_half_win_samples = int(kurt_window / 2)
        start_idx_in_segment = max(0, start_idx_in_segment - kurt_half_win_samples)
        end_idx_in_segment = min(len(time_segment) - 1, end_idx_in_segment + kurt_half_win_samples)

        start_time = time_segment[start_idx_in_segment]
        end_time = time_segment[end_idx_in_segment]


        # Expand slightly and ensure minimum duration
        duration = end_time - start_time
        expansion = max(0, (min_duration - duration) / 2.0) # Expand to meet min_duration

        final_start_time = start_time - expansion
        final_end_time = end_time + expansion

        # Ensure we don't exceed original scan boundaries
        final_start_time = max(start_scan, final_start_time)
        final_end_time = min(end_scan, final_end_time)

        # Final check: Ensure end > start and handle edge cases
        if final_end_time <= final_start_time:
             # If detection failed or resulted in invalid range, revert to scan range
             final_start_time = start_scan
             final_end_time = end_scan
        elif final_end_time - final_start_time < 0.01: # If range is too small, revert
             final_start_time = start_scan
             final_end_time = end_scan


        # print(f"Shock detected: Kurtosis range [{first_shock_time_kurt:.2f}s, {last_shock_time_kurt:.2f}s] -> Signal range [{final_start_time:.2f}s, {final_end_time:.2f}s]")
        return final_start_time, final_end_time
    else:
        # print(f"No significant shock detected via kurtosis in scan range [{start_scan}s, {end_scan}s]. Using full range.")
        return start_scan, end_scan # Return original range if no shock

def process_file(file_path, maturity_label, domain_label,
                 feature_type, # Added feature_type ('cnn' or 'handcrafted')
                 expected_length, # Only needed for CNN
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
        # print(f"Warning: No data in determined range [{start_time:.2f}s, {end_time:.2f}s] for {file_path}. Skipping.")
        return [], [], [], []

    # Create original sample + augmentations
    all_signals_for_file = [signal_segment] + augment_signal(signal_segment, n_augments=n_augments)

    handcrafted_list = []
    processed_signals_list = []
    maturity_labels_list = []
    domain_labels_list = []

    for sig in all_signals_for_file:
        if len(sig) == 0: continue # Skip empty augmented signals if they occur

        processed_signal_for_cnn = np.array([])
        # --- Handle signal length difference ONLY IF USING CNN---
        if feature_type == 'cnn':
            if expected_length is None or expected_length <= 0:
                 raise ValueError("Expected signal length must be positive for CNN features.")
            current_length = len(sig)
            if current_length != expected_length:
                if current_length < expected_length:
                    pad_width = expected_length - current_length
                    # Use reflection padding (or 'edge' might be simpler/faster)
                    sig_padded = np.pad(sig, (0, pad_width), 'reflect')
                else:
                    # Truncate (from the end is simplest)
                    sig_padded = sig[:expected_length]
            else:
                sig_padded = sig # Already correct length
            processed_signal_for_cnn = sig_padded
        else:
             # For handcrafted, we use the original segmented signal 'sig'
             pass # No padding/truncation needed for handcrafted feature extraction stage

        # --- Feature Extraction ---
        # Always extract handcrafted, but only use them if feature_type is handcrafted
        features_hc = extract_handcrafted_features(sig) # Extract from the segment before padding/truncating
        handcrafted_list.append(features_hc)

        # Store the potentially padded/truncated signal only if using CNN
        if feature_type == 'cnn':
            processed_signals_list.append(processed_signal_for_cnn)
        else:
            processed_signals_list.append(np.array([])) # Append empty array if not using CNN signals


        maturity_labels_list.append(maturity_label) # 0 for ripe, 1 for rotten
        domain_labels_list.append(domain_label)

    # Decide what signal data to return based on feature_type
    if feature_type == 'cnn':
        return_signals = processed_signals_list
    else:
        # If using handcrafted, we don't need to return the bulky signal data unless for debugging
        # Return empty arrays for signals to save memory, handcrafted features are the primary output
         return_signals = [np.array([])] * len(handcrafted_list)


    return handcrafted_list, return_signals, maturity_labels_list, domain_labels_list


def collect_data_from_folders(fruit_name, ripe_folder, rotten_folder, domain_label,
                              feature_type, # Added feature_type
                              expected_length, # Added expected_length
                              cfg, # Pass config object
                              n_augments=None): # Allow override, default to config
    """Collects and processes data from ripe and rotten folders for a fruit."""
    all_handcrafted = []
    all_signals = [] # Will contain padded/truncated signals if CNN, else potentially empty arrays
    all_maturity_labels = []
    all_domain_labels = []

    if n_augments is None:
        n_augments = cfg.AUGMENTATIONS_PER_IMAGE

    print(f"Processing {fruit_name} (Domain {domain_label})...")
    # Process Ripe (Label 0)
    print(f"  Processing Ripe folder: {ripe_folder}")
    ripe_files = [f for f in os.listdir(ripe_folder) if f.endswith('.csv')]
    num_processed_ripe = 0
    for f in ripe_files:
        file_path = os.path.join(ripe_folder, f)
        h, s, m, d = process_file(
            file_path, 0, domain_label,
            feature_type=feature_type,
            expected_length=expected_length,
            start_scan=cfg.SIGNAL_START_TIME,
            end_scan=cfg.SIGNAL_END_TIME,
            freq=cfg.SAMPLING_FREQUENCY,
            n_augments=n_augments
        )
        if h: # Check if processing returned any data
            all_handcrafted.extend(h)
            all_signals.extend(s)
            all_maturity_labels.extend(m)
            all_domain_labels.extend(d)
            num_processed_ripe += len(h)

    print(f"    Found {len(ripe_files)} ripe files -> {num_processed_ripe} samples (incl. augmentations).")


    # Process Rotten (Label 1)
    # rotten_start_idx = len(all_handcrafted) # Keep track for stats (alternative way)
    print(f"  Processing Rotten folder: {rotten_folder}")
    rotten_files = [f for f in os.listdir(rotten_folder) if f.endswith('.csv')]
    num_processed_rotten = 0
    for f in rotten_files:
        file_path = os.path.join(rotten_folder, f)
        h, s, m, d = process_file(
             file_path, 1, domain_label,
             feature_type=feature_type,
             expected_length=expected_length,
             start_scan=cfg.SIGNAL_START_TIME,
             end_scan=cfg.SIGNAL_END_TIME,
             freq=cfg.SAMPLING_FREQUENCY,
             n_augments=n_augments
        )
        if h: # Check if processing returned any data
            all_handcrafted.extend(h)
            all_signals.extend(s)
            all_maturity_labels.extend(m)
            all_domain_labels.extend(d)
            num_processed_rotten += len(h)
    # rotten_count = len(all_handcrafted) - rotten_start_idx
    print(f"    Found {len(rotten_files)} rotten files -> {num_processed_rotten} samples (incl. augmentations).")

    if not all_handcrafted: # Check if any data was processed
        print(f"Warning: No data collected for {fruit_name}. Check paths and file contents.")
        return np.array([]), np.array([]), np.array([]), np.array([])

    # Convert lists to numpy arrays
    all_handcrafted_np = np.array(all_handcrafted, dtype=np.float32)
    all_maturity_labels_np = np.array(all_maturity_labels, dtype=np.int64)
    all_domain_labels_np = np.array(all_domain_labels, dtype=np.int64)

    # Handle signals carefully - they might be empty lists if not CNN
    if feature_type == 'cnn':
        # Ensure all signals have the same length before stacking
        try:
             all_signals_np = np.array(all_signals, dtype=np.float32)
             if all_signals_np.ndim == 1: # If it stacked incorrectly (e.g., list of arrays of different len)
                 print("Warning: Signal lengths might be inconsistent. Check preprocessing.")
                 # Find max length and pad? Or error? Let's assume process_file handles it.
                 # If process_file ensures consistent length, stacking should work.
                 pass # Assume lengths are correct based on process_file logic
        except ValueError as e:
             print(f"Error converting signals to numpy array (likely inconsistent lengths): {e}")
             print("Returning empty signal array. Handcrafted features will still be used.")
             # Return empty array for signals if conversion fails
             all_signals_np = np.array([])

    else: # Handcrafted feature type
        all_signals_np = np.array([]) # Explicitly return empty array


    return (all_handcrafted_np, all_signals_np,
            all_maturity_labels_np, all_domain_labels_np)


if __name__ == '__main__':
    # Example Usage (demonstration requires a dummy config)
    print("Demonstrating data collection...")

    # Create a dummy config object for demonstration
    class DummyConfig:
        FEATURE_TYPE = 'handcrafted' # Or 'cnn'
        SIGNAL_START_TIME = 6.0
        SIGNAL_END_TIME = 8.1
        SAMPLING_FREQUENCY = 250
        AUGMENTATIONS_PER_IMAGE = 1
        HANDCRAFTED_DIM = 5
        if FEATURE_TYPE == 'cnn':
            EXPECTED_SIGNAL_LENGTH = calculate_expected_length(SIGNAL_START_TIME, SIGNAL_END_TIME, SAMPLING_FREQUENCY)
        else:
            EXPECTED_SIGNAL_LENGTH = None # Not strictly needed for model if handcrafted

    cfg = DummyConfig()

    # --- Define dummy paths for demonstration ---
    dummy_base = "dummy_data_demo"
    os.makedirs(os.path.join(dummy_base, "grape_ripe"), exist_ok=True)
    os.makedirs(os.path.join(dummy_base, "grape_rotten"), exist_ok=True)
    # Create dummy CSV files
    time_dummy = np.linspace(0, 10, 10 * cfg.SAMPLING_FREQUENCY) # Longer time series
    force_dummy_ripe = np.sin(time_dummy * 5) + np.random.randn(len(time_dummy)) * 0.1
    # Add a 'shock' within the scan range
    shock_start_idx = int(cfg.SIGNAL_START_TIME * cfg.SAMPLING_FREQUENCY + 50)
    shock_end_idx = int(shock_start_idx + 0.2 * cfg.SAMPLING_FREQUENCY)
    force_dummy_ripe[shock_start_idx:shock_end_idx] += 5
    force_dummy_rotten = np.sin(time_dummy * 2) + np.random.randn(len(time_dummy)) * 0.2 # No explicit shock
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
        feature_type=cfg.FEATURE_TYPE,
        expected_length=cfg.EXPECTED_SIGNAL_LENGTH,
        cfg=cfg # Pass the dummy config
        # n_augments will default to cfg.AUGMENTATIONS_PER_IMAGE
    )

    print("\nCollected Data Shapes:")
    print(f"  Handcrafted Features: {h_feat.shape}")      # (num_samples, num_handcrafted_features)
    if cfg.FEATURE_TYPE == 'cnn':
        print(f"  Signals:            {signals.shape}")       # (num_samples, expected_length)
    else:
        print(f"  Signals:            Collected {len(signals)} arrays (expected empty for handcrafted type)") # Should be empty or list of empty arrays
    print(f"  Maturity Labels:    {mat_labels.shape}")    # (num_samples,) -> Binary (0 or 1)
    print(f"  Domain Labels:      {dom_labels.shape}")     # (num_samples,) -> Domain index (0, 1, 2...)

    if h_feat.size > 0:
        print("\nSample Data:")
        print("  Handcrafted Features (first sample):\n", h_feat[0])
        print("  Maturity Label (first sample):", mat_labels[0])
        print("  Domain Label (first sample):", dom_labels[0])

        # Plot the first signal segment *if collected and not empty*
        if cfg.FEATURE_TYPE == 'cnn' and signals.size > 0 and len(signals[0]) > 0:
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
    import shutil
    # Be careful with rmtree!
    # try:
    #     shutil.rmtree(dummy_base)
    #     print(f"Removed dummy directory: {dummy_base}")
    # except OSError as e:
    #     print(f"Error removing dummy directory {dummy_base}: {e}")