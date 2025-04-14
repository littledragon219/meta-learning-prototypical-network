# test_p1_p2.py
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib
import time

# Import from local modules
import config # Load config
from datapreprocessing import collect_data_from_folders, calculate_expected_length
from model import (
    FeatureExtractor1DCNN, HandcraftedFeatureExtractor, MaturityClassifier
)
from utils import (
    calculate_prototypes, prototypical_loss, # Need loss logic for classification distance calc
    plot_confusion_matrix_heatmap, plot_tsne
)
# Import helper functions defined in train_phase2_dg (or copy them here)
try:
    from train_phase2_dg import get_combined_label, get_combined_class_names
except ImportError:
    print("Warning: Could not import helper functions from train_phase2_dg. Redefining them.")
    def get_combined_label(domain_idx, maturity_idx, num_maturity_classes=2):
        return domain_idx * num_maturity_classes + maturity_idx
    def get_combined_class_names(domain_map, maturity_names=["Ripe", "Rotten"]):
        names = []
        num_maturity = len(maturity_names)
        for domain_idx in sorted(domain_map.keys()):
            domain_name = domain_map[domain_idx]
            for mat_idx, mat_name in enumerate(maturity_names):
                names.append(f"{domain_name}-{mat_name}")
        return names

def load_test_data(cfg):
    """Loads and preprocesses the test data."""
    print("--- Loading Test Data ---")
    all_h_feat_test, all_signals_test, all_mat_labels_test, all_dom_labels_test = [], [], [], []
    test_domain_mapping = {}
    test_domain_idx_counter = 0
    expected_len = None
    if cfg.FEATURE_TYPE == 'cnn': expected_len = calculate_expected_length(cfg.SIGNAL_START_TIME, cfg.SIGNAL_END_TIME, cfg.SAMPLING_FREQUENCY)

    for fruit_name, paths in cfg.TEST_DATA_PATHS.items():
        if fruit_name not in test_domain_mapping: test_domain_mapping[fruit_name] = test_domain_idx_counter; test_domain_idx_counter += 1
        current_domain_idx = test_domain_mapping[fruit_name]
        print(f"Loading test data for: {fruit_name} (Domain Index: {current_domain_idx})")
        h_feat, signals, mat_labels, dom_labels = collect_data_from_folders(
            fruit_name=fruit_name, ripe_folder=paths["ripe"], rotten_folder=paths["rotten"], domain_label=current_domain_idx,
            feature_type=cfg.FEATURE_TYPE, expected_length=expected_len, cfg=cfg, n_augments=0 )
        if h_feat.size > 0:
            all_h_feat_test.append(h_feat); all_signals_test.append(signals); all_mat_labels_test.append(mat_labels); all_dom_labels_test.append(dom_labels)
        else: print(f"Warning: No test data loaded for {fruit_name}.")
    if not all_h_feat_test: raise ValueError("No test data collected. Check TEST_DATA_PATHS.")

    h_feat_all_test = np.concatenate(all_h_feat_test, axis=0)
    mat_labels_all_test = np.concatenate(all_mat_labels_test, axis=0)
    dom_labels_all_test = np.concatenate(all_dom_labels_test, axis=0)
    if cfg.FEATURE_TYPE == 'cnn':
        valid_signals = [s for s in all_signals_test if s is not None and s.size > 0]
        if valid_signals: signals_all_test = np.concatenate(valid_signals, axis=0)
        else: raise ValueError("CNN mode selected, but no valid test signal data was loaded.")
        if signals_all_test.shape[0] != h_feat_all_test.shape[0]: raise ValueError("Test signal/label count mismatch.")
    else: signals_all_test = np.array([])

    scaler = None
    X_test_scaled = h_feat_all_test
    if config.FEATURE_TYPE == 'handcrafted':
        try: scaler = joblib.load(config.P1_SCALER_PATH); X_test_scaled = scaler.transform(h_feat_all_test); print(f"Loaded/applied Phase 1 scaler.")
        except FileNotFoundError: print(f"Error: Scaler not found."); raise
        except Exception as e: print(f"Error loading/applying scaler: {e}"); raise

    if config.FEATURE_TYPE == 'cnn': X_test_tensor = torch.tensor(signals_all_test, dtype=torch.float32)
    else: X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_domain_test_tensor = torch.tensor(dom_labels_all_test, dtype=torch.long)
    y_maturity_test_tensor = torch.tensor(mat_labels_all_test, dtype=torch.long)
    test_dataset = TensorDataset(X_test_tensor, y_domain_test_tensor, y_maturity_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=config.PHASE1_BATCH_SIZE, shuffle=False, num_workers=0)
    print(f"Test dataset created: {len(test_dataset)} samples.")
    test_domain_reverse_mapping = {v: k for k, v in test_domain_mapping.items()}
    return test_loader, test_domain_reverse_mapping

def evaluate_domain_classification(model, prototypes, test_loader, device, domain_map):
    """Evaluates domain classification using features and prototypes."""
    model.eval()
    all_features, all_true_dom, all_pred_dom = [], [], []
    with torch.no_grad():
        for inputs, true_domains, _ in test_loader:
            inputs, true_domains = inputs.to(device), true_domains.to(device)
            features = model(inputs)
            if prototypes is None: print("Error: Prototypes missing."); return None, None, None, 0.0
            prototypes = prototypes.to(device)
            query_norm = F.normalize(features, p=2, dim=1)
            proto_norm = F.normalize(prototypes, p=2, dim=1); proto_norm = torch.nan_to_num(proto_norm)
            logits = torch.mm(query_norm, proto_norm.t())
            preds = torch.argmax(logits, dim=1)
            all_features.append(features.cpu()); all_true_dom.extend(true_domains.cpu().numpy()); all_pred_dom.extend(preds.cpu().numpy())
    if not all_true_dom: print("Warning: No domain results."); return None, None, None, 0.0
    all_features = torch.cat(all_features).numpy()
    print("\n--- Domain Classification Report ---")
    labels_list = sorted(domain_map.keys()); target_names_list=[domain_map.get(i, f"Unknown_{i}") for i in labels_list]
    report = classification_report(all_true_dom, all_pred_dom, labels=labels_list, target_names=target_names_list, zero_division=0); print(report)
    accuracy = accuracy_score(all_true_dom, all_pred_dom); print(f"Domain Classification Accuracy: {accuracy:.4f}")
    return all_true_dom, all_pred_dom, all_features, accuracy

def evaluate_combined_prediction(p2_model, p2_prototypes, s2_maturity_models, test_loader, device, domain_map):
    """Evaluates the two-step prediction: predict domain, then predict maturity."""
    p2_model.eval()
    all_true_dom, all_true_mat = [], []
    all_pred_dom_step1, all_pred_mat_step2 = [], []
    all_p2_features = [] # Collect features for t-SNE
    missing_s2_warnings = set()

    with torch.no_grad():
        for inputs, true_domains, true_maturity in test_loader:
            inputs = inputs.to(device)
            true_domains_np = true_domains.numpy()
            true_maturity_np = true_maturity.numpy()
            all_true_dom.extend(true_domains_np)
            all_true_mat.extend(true_maturity_np)

            # Step 1: Get P2 features and predict domain
            features_p2 = p2_model(inputs)
            all_p2_features.append(features_p2.cpu()) # For t-SNE

            if p2_prototypes is None: print("Error: P2 Prototypes missing."); return None, None, None, None, None, 0.0
            prototypes = p2_prototypes.to(device)
            query_norm = F.normalize(features_p2, p=2, dim=1)
            proto_norm = F.normalize(prototypes, p=2, dim=1); proto_norm = torch.nan_to_num(proto_norm)
            logits_dom = torch.mm(query_norm, proto_norm.t())
            predicted_domains = torch.argmax(logits_dom, dim=1)
            all_pred_dom_step1.extend(predicted_domains.cpu().numpy())

            # Step 2: Predict maturity using predicted domain's S2 head
            batch_preds_maturity = []
            for i in range(features_p2.shape[0]):
                sample_feature = features_p2[i:i+1]
                pred_dom_idx = predicted_domains[i].item()

                if pred_dom_idx in s2_maturity_models:
                    s2_model = s2_maturity_models[pred_dom_idx]
                    s2_model.eval()
                    maturity_logits = s2_model(sample_feature)
                    pred_mat = torch.argmax(maturity_logits, dim=1).item()
                    batch_preds_maturity.append(pred_mat)
                else:
                    if pred_dom_idx not in missing_s2_warnings:
                         print(f"Warning: S2 model missing for predicted domain {pred_dom_idx} ({domain_map.get(pred_dom_idx, 'Unknown')}). Predicting 0.")
                         missing_s2_warnings.add(pred_dom_idx)
                    batch_preds_maturity.append(0)
            all_pred_mat_step2.extend(batch_preds_maturity)

    if not all_true_mat: print("Warning: No combined results."); return None, None, None, None, None, 0.0

    all_p2_features = torch.cat(all_p2_features).numpy()

    # --- Evaluate Step 1 (Domain Prediction Accuracy) ---
    print("\n--- Domain Classification Report (P2 Model - Test Data) ---")
    labels_list = sorted(domain_map.keys()); target_names_list=[domain_map.get(i, f"Unknown_{i}") for i in labels_list]
    report_dom = classification_report(all_true_dom, all_pred_dom_step1, labels=labels_list, target_names=target_names_list, zero_division=0); print(report_dom)
    accuracy_dom = accuracy_score(all_true_dom, all_pred_dom_step1); print(f"Test Domain Classification Accuracy (P2 Model): {accuracy_dom:.4f}")

    # --- Evaluate Step 2 (Combined Prediction Accuracy) ---
    print("\n--- Combined Prediction Report (P2 Model + S2 Heads - Test Data) ---")
    num_mat_classes = 2 # Ripe, Rotten
    true_combined = [get_combined_label(d, m, num_mat_classes) for d, m in zip(all_true_dom, all_true_mat)]
    pred_combined = [get_combined_label(d, m, num_mat_classes) for d, m in zip(all_pred_dom_step1, all_pred_mat_step2)] # Use predicted domain + predicted maturity
    combined_names = get_combined_class_names(domain_map)
    combined_labels_list = list(range(len(combined_names)))

    report_comb = classification_report(true_combined, pred_combined, labels=combined_labels_list, target_names=combined_names, zero_division=0); print(report_comb)
    accuracy_comb = accuracy_score(true_combined, pred_combined); print(f"Test Combined Prediction Accuracy: {accuracy_comb:.4f}")

    return all_true_dom, all_pred_dom_step1, true_combined, pred_combined, all_p2_features, accuracy_comb


def main():
    print("--- Starting P1 & P2 Model Evaluation (Revised Logic) ---")
    start_time = time.time()
    os.makedirs(config.TEST_RESULTS_DIR, exist_ok=True)

    # --- 1. Load Test Data ---
    try: test_loader, test_domain_map = load_test_data(config)
    except Exception as e: print(f"Failed to load test data: {e}"); return
    num_test_domains = len(test_domain_map)
    test_domain_names = [test_domain_map.get(i) for i in sorted(test_domain_map.keys())]

    # --- 2. Evaluate Phase 1 Model (Domain Classification Only) ---
    print("\n=== Evaluating Phase 1 Model (Domain Classification) ===")
    try:
        expected_len_cnn = None
        if config.FEATURE_TYPE == 'cnn': expected_len_cnn = calculate_expected_length(config.SIGNAL_START_TIME, config.SIGNAL_END_TIME, config.SAMPLING_FREQUENCY)
        if config.FEATURE_TYPE == 'cnn': p1_model = FeatureExtractor1DCNN(sequence_length=expected_len_cnn, embedding_dim=config.EMBEDDING_DIM).to(config.DEVICE)
        else: p1_model = HandcraftedFeatureExtractor(input_dim=config.HANDCRAFTED_DIM, embedding_dim=config.EMBEDDING_DIM).to(config.DEVICE)
        p1_model.load_state_dict(torch.load(config.P1_MODEL_SAVE_PATH, map_location=config.DEVICE)); print(f"Loaded P1 model.")
        p1_prototypes = torch.load(config.P1_PROTOTYPES_SAVE_PATH, map_location=config.DEVICE)
        if p1_prototypes.shape[0] != num_test_domains: print(f"Warning: P1 prototypes ({p1_prototypes.shape[0]}) != num test domains ({num_test_domains}).")
        print(f"Loaded P1 prototypes.")
        p1_true_dom, p1_pred_dom, p1_features, p1_acc = evaluate_domain_classification(p1_model, p1_prototypes, test_loader, config.DEVICE, test_domain_map)
        if p1_true_dom is not None:
            plot_confusion_matrix_heatmap(p1_true_dom, p1_pred_dom, test_domain_names, 'Phase 1 Test CM (Domains)', config.P1_TEST_DOMAIN_CONFMAT_PATH)
            plot_tsne(p1_features, p1_true_dom, test_domain_map, 'Phase 1 Test t-SNE (True Domain)', config.P1_TEST_TSNE_PATH)
    except FileNotFoundError as e: print(f"Error loading P1 artifacts: {e}. Skipping.")
    except Exception as e: print(f"Error during P1 evaluation: {e}")

    # --- 3. Evaluate Phase 2 Model (Combined Prediction) ---
    print("\n=== Evaluating Phase 2 Model (Combined Domain + Maturity Prediction) ===")
    try:
        expected_len_cnn = None
        if config.FEATURE_TYPE == 'cnn': expected_len_cnn = calculate_expected_length(config.SIGNAL_START_TIME, config.SIGNAL_END_TIME, config.SAMPLING_FREQUENCY)
        if config.FEATURE_TYPE == 'cnn': p2_model = FeatureExtractor1DCNN(sequence_length=expected_len_cnn, embedding_dim=config.EMBEDDING_DIM).to(config.DEVICE)
        else: p2_model = HandcraftedFeatureExtractor(input_dim=config.HANDCRAFTED_DIM, embedding_dim=config.EMBEDDING_DIM).to(config.DEVICE)
        p2_model.load_state_dict(torch.load(config.P2_MODEL_SAVE_PATH, map_location=config.DEVICE)); print(f"Loaded P2 model.")
        p2_prototypes = torch.load(config.P2_PROTOTYPES_SAVE_PATH, map_location=config.DEVICE)
        if p2_prototypes.shape[0] != num_test_domains: print(f"Warning: P2 prototypes ({p2_prototypes.shape[0]}) != num test domains ({num_test_domains}).")
        print(f"Loaded P2 prototypes.")

        s2_maturity_models = {}
        for domain_idx, domain_name in test_domain_map.items():
            s2_model_path = config.S2_MODEL_SAVE_TEMPLATE.format(domain_name)
            if os.path.exists(s2_model_path):
                try:
                    model_s2 = MaturityClassifier(input_dim=config.EMBEDDING_DIM, num_classes=2).to(config.DEVICE)
                    model_s2.load_state_dict(torch.load(s2_model_path, map_location=config.DEVICE)); model_s2.eval()
                    s2_maturity_models[domain_idx] = model_s2; print(f"  Loaded S2 model for {domain_name}")
                except Exception as e: print(f"  Error loading S2 model for {domain_name}: {e}")
            else: print(f"  Warning: S2 model for {domain_name} not found.")
        if len(s2_maturity_models) != num_test_domains: print("Warning: Not all required S2 models were loaded for test evaluation.")

        # Evaluate combined prediction
        p2_true_dom, p2_pred_dom, p2_true_comb, p2_pred_comb, p2_features, p2_comb_acc = evaluate_combined_prediction(
            p2_model, p2_prototypes, s2_maturity_models, test_loader, config.DEVICE, test_domain_map
        )

        # Visualize P2 Results
        if p2_true_dom is not None:
            plot_confusion_matrix_heatmap(p2_true_dom, p2_pred_dom, test_domain_names, 'Phase 2 Test CM (Domains - Step 1)', config.P2_TEST_DOMAIN_CONFMAT_PATH)
        if p2_true_comb is not None:
            combined_names = get_combined_class_names(test_domain_map)
            plot_confusion_matrix_heatmap(p2_true_comb, p2_pred_comb, combined_names, 'Phase 2 Test CM (Combined Domain-Maturity)', config.P2_TEST_COMBINED_CONFMAT_PATH)
        if p2_features is not None and p2_true_dom is not None:
             plot_tsne(p2_features, p2_true_dom, test_domain_map, 'Phase 2 Test t-SNE (True Domain)', config.P2_TEST_TSNE_PATH)

    except FileNotFoundError as e: print(f"Error loading P2/S2 artifacts: {e}. Skipping.")
    except Exception as e: print(f"Error during P2 evaluation: {e}")

    end_time = time.time()
    print(f"\n--- Test Script Completed in {end_time - start_time:.2f} seconds ---")

if __name__ == "__main__":
    main()