# inference_phase3_zsl.py
import os
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix # Added imports
import joblib
import time

# Import from local modules
import config # Load config
from datapreprocessing import collect_data_from_folders, calculate_expected_length
from model import (
    FeatureExtractor1DCNN, HandcraftedFeatureExtractor # MaturityClassifier not used here
)
from utils import (
     plot_confusion_matrix_heatmap, plot_tsne, calculate_prototypes, prototypical_loss # Reusing utils
)
# Import helper function defined in train_phase1_base (ensure it's runnable standalone if needed)
try:
    from train_phase1_base import get_subprototype_mapping # Needed for loading/interpreting subprotos
except ImportError:
    print("Warning: Could not import get_subprototype_mapping from train_phase1_base.")
    # Define a fallback or ensure train_phase1_base.py is accessible
    def get_subprototype_mapping(domain_map, maturity_map={"ripe": 0, "rotten": 1}):
        # Basic fallback implementation
        subproto_map = {}
        num_maturity = len(maturity_map)
        mat_idx_to_name = {v: k for k, v in maturity_map.items()}
        for domain_name, domain_idx in domain_map.items():
            for mat_name, mat_idx in maturity_map.items():
                combined_idx = domain_idx * num_maturity + mat_idx
                subproto_map[combined_idx] = {
                    "name": f"{domain_name}-{mat_name}",
                    "domain_idx": domain_idx,
                    "maturity_idx": mat_idx,
                    "maturity_name": mat_name
                }
        return subproto_map

def load_inference_artifacts_revised(cfg):
    """
    Loads artifacts needed for Phase 3 inference:
    - P1 Scaler (if handcrafted)
    - P2 Feature Extractor (DG model)
    - P2 Domain Prototypes (for domain prediction)
    - P1 Maturity Sub-Prototypes (for maturity prediction) + Mapping
    """
    print("--- Loading Artifacts for Phase 3 Inference (Revised Two-Step) ---")
    scaler = None
    if cfg.FEATURE_TYPE == 'handcrafted':
        try:
            scaler = joblib.load(cfg.P1_SCALER_PATH)
            print(f"Loaded P1 scaler from {cfg.P1_SCALER_PATH}")
        except Exception as e: print(f"Error loading P1 scaler: {e}."); raise

    print("Loading Phase 2 feature extractor (DG model)...")
    expected_len_cnn = None
    if cfg.FEATURE_TYPE == 'cnn':
        expected_len_cnn = calculate_expected_length(cfg.SIGNAL_START_TIME, cfg.SIGNAL_END_TIME, cfg.SAMPLING_FREQUENCY)
    if cfg.FEATURE_TYPE == 'cnn':
        model_p2_extractor = FeatureExtractor1DCNN(sequence_length=expected_len_cnn, embedding_dim=cfg.EMBEDDING_DIM).to(cfg.DEVICE)
    else: # handcrafted
        model_p2_extractor = HandcraftedFeatureExtractor(input_dim=cfg.HANDCRAFTED_DIM, embedding_dim=cfg.EMBEDDING_DIM).to(cfg.DEVICE)
    try:
        model_p2_extractor.load_state_dict(torch.load(cfg.P2_MODEL_SAVE_PATH, map_location=cfg.DEVICE))
        model_p2_extractor.eval()
        print(f"Loaded P2 DG model from {cfg.P2_MODEL_SAVE_PATH}")
    except Exception as e: print(f"Error loading P2 model: {e}"); raise

    print("Loading Phase 2 known domain prototypes...")
    try:
        # These are prototypes calculated from P2 training data (Grape, Strawberry)
        known_domain_prototypes_p2 = torch.load(cfg.P2_PROTOTYPES_SAVE_PATH, map_location=cfg.DEVICE)
        known_domain_mapping_p2 = {name: i for i, name in enumerate(config.PHASE2_DATA_PATHS.keys())}
        num_known_domains_p2 = len(known_domain_mapping_p2)
        if known_domain_prototypes_p2.shape[0] != num_known_domains_p2:
            print(f"Warning: P2 Prototypes count ({known_domain_prototypes_p2.shape[0]}) != P2 domains ({num_known_domains_p2}). Trusting loaded count.")
            num_known_domains_p2 = known_domain_prototypes_p2.shape[0] # Adjust based on loaded file
        print(f"Loaded {num_known_domains_p2} P2 known domain prototypes from {cfg.P2_PROTOTYPES_SAVE_PATH}")
        known_domain_reverse_map_p2 = {v: k for k, v in known_domain_mapping_p2.items() if v < num_known_domains_p2}
    except Exception as e: print(f"Error loading P2 prototypes: {e}"); raise

    print("Loading Phase 1 maturity sub-prototypes...")
    try:
        subproto_data = torch.load(cfg.P1_SUBPROTOTYPES_SAVE_PATH, map_location=cfg.DEVICE)
        sub_prototypes = subproto_data['prototypes'].to(cfg.DEVICE) # Ensure on correct device
        sub_prototype_mapping = subproto_data['mapping']
        num_sub_protos = sub_prototypes.shape[0]
        print(f"Loaded {num_sub_protos} sub-prototypes from {cfg.P1_SUBPROTOTYPES_SAVE_PATH}")
        print(f"  Sub-prototype mapping loaded.")
    except Exception as e: print(f"Error loading P1 sub-prototypes: {e}"); raise

    # Mapping for known domains from Phase 1 (should match Phase 2)
    known_domain_mapping_p1 = {name: i for i, name in enumerate(config.PHASE1_DATA_PATHS.keys())}

    # Check consistency between P1 and P2 domain mappings if needed
    if known_domain_mapping_p1 != known_domain_mapping_p2:
        print("Warning: Domain mappings between Phase 1 and Phase 2 config differ.")
        # Decide which mapping to trust or reconcile, P2 mapping is likely more relevant for P2 model/protos
    domain_map_for_inference = known_domain_reverse_map_p2 # Use the mapping relevant to the loaded P2 prototypes

    return (scaler, model_p2_extractor, known_domain_prototypes_p2, sub_prototypes,
            sub_prototype_mapping, domain_map_for_inference)


def run_zsl_inference_two_step_subproto():
    # --- Phase 3: Test on Tomato ---
    print(f"--- Starting Phase 3: ZSL Inference on '{config.PHASE3_NEW_FRUIT_NAME}' (Two-Step Sub-Proto) ---")
    start_time_p3 = time.time()
    os.makedirs(config.BASE_SAVE_DIR, exist_ok=True)

    # 1. Load Artifacts
    try:
        scaler, model_extractor, known_domain_protos, \
        sub_prototypes, sub_prototype_mapping, \
        known_domain_reverse_map = load_inference_artifacts_revised(config)
    except Exception as e:
        print(f"Failed to load necessary artifacts for Phase 3. Aborting. Error: {e}")
        return

    num_known_domains = known_domain_protos.shape[0]
    # Assign the next available index to the new domain (Tomato)
    new_domain_idx = num_known_domains
    domain_names_all_map_reverse = {**known_domain_reverse_map, new_domain_idx: config.PHASE3_NEW_FRUIT_NAME}
    domain_names_all_map = {v: k for k, v in domain_names_all_map_reverse.items()} # name -> index
    print(f"\nDomain mapping for inference: {domain_names_all_map}")

    # --- 2. Load and Prepare New Fruit Data (Tomato) ---
    print(f"\nLoading data for new fruit: '{config.PHASE3_NEW_FRUIT_NAME}'")
    expected_len = None
    if config.FEATURE_TYPE == 'cnn':
        expected_len = calculate_expected_length(config.SIGNAL_START_TIME, config.SIGNAL_END_TIME, config.SAMPLING_FREQUENCY)

    h_feat_new, signals_new, mat_labels_new, _ = collect_data_from_folders(
        fruit_name=config.PHASE3_NEW_FRUIT_NAME,
        ripe_folder=config.PHASE3_NEW_FRUIT_PATHS["ripe"],
        rotten_folder=config.PHASE3_NEW_FRUIT_PATHS["rotten"],
        domain_label=new_domain_idx, # Assign the new domain index
        feature_type=config.FEATURE_TYPE, expected_length=expected_len, cfg=config, n_augments=0
    )
    if h_feat_new.size == 0: print(f"Error: No data for '{config.PHASE3_NEW_FRUIT_NAME}'."); return
    print(f"Loaded {len(mat_labels_new)} samples for '{config.PHASE3_NEW_FRUIT_NAME}'.")

    # Prepare input tensor
    if config.FEATURE_TYPE == 'cnn':
        X_new_tensor = torch.tensor(signals_new, dtype=torch.float32).to(config.DEVICE)
    else: # handcrafted
        if scaler is None: print("Error: Scaler not loaded."); return
        h_feat_new_scaled = scaler.transform(h_feat_new)
        X_new_tensor = torch.tensor(h_feat_new_scaled, dtype=torch.float32).to(config.DEVICE)

    # True labels
    y_true_m_new = torch.tensor(mat_labels_new, dtype=torch.long).numpy()
    y_true_d_new = torch.full((len(y_true_m_new),), new_domain_idx, dtype=torch.long).numpy()

    # --- 3. Extract Features using P2 Model ---
    print("Extracting features for the new fruit...")
    model_extractor.eval()
    with torch.no_grad():
        new_fruit_features = model_extractor(X_new_tensor) # (N_new, embedding_dim)
    print(f"Extracted features shape: {new_fruit_features.shape}")

    # --- 4. Initialize ZSL Domain Prototype for Tomato ---
    print("Initializing ZSL DOMAIN prototype for the new fruit (direct mean)...")
    if new_fruit_features.numel() > 0:
        zsl_domain_prototype = new_fruit_features.mean(dim=0, keepdim=True).to(config.DEVICE)
    else:
        print("Warning: No features for ZSL domain prototype. Using zeros."); zsl_domain_prototype = torch.zeros(1, config.EMBEDDING_DIM, device=config.DEVICE)

    # --- 5. Combine Domain Prototypes ---
    all_domain_prototypes_for_inference = torch.cat([known_domain_protos, zsl_domain_prototype], dim=0)
    num_total_domains_inference = all_domain_prototypes_for_inference.shape[0]
    print(f"Total DOMAIN prototypes for inference: {num_total_domains_inference}")

    # --- 6. Step 1: Predict Domain for New Fruit Samples ---
    print(f"\nStep 1: Predicting Domain for '{config.PHASE3_NEW_FRUIT_NAME}' Samples...")
    pred_domains_new = []
    model_extractor.eval()
    with torch.no_grad():
        # Use cosine similarity for domain prediction consistency
        query_norm = F.normalize(new_fruit_features, p=2, dim=1)
        proto_norm = F.normalize(all_domain_prototypes_for_inference, p=2, dim=1)
        proto_norm = torch.nan_to_num(proto_norm)
        similarities_dom = torch.mm(query_norm, proto_norm.t()) # (N_new, N_total_domains)
        pred_domains_new_tensor = torch.argmax(similarities_dom, dim=1)
        pred_domains_new = pred_domains_new_tensor.cpu().numpy() # Store predicted domain indices

    # --- Evaluate Step 1 (Domain Classification) ---
    print(f"\n--- Domain Classification Results for '{config.PHASE3_NEW_FRUIT_NAME}' Samples ---")
    all_domain_names_list = [domain_names_all_map_reverse.get(i, f'Unknown_{i}') for i in sorted(domain_names_all_map_reverse.keys())]
    report_domain = classification_report(y_true_d_new, pred_domains_new,
                                labels=list(domain_names_all_map_reverse.keys()),
                                target_names=all_domain_names_list, zero_division=0)
    print(report_domain)
    zsl_domain_accuracy = accuracy_score(y_true_d_new, pred_domains_new)
    print(f"Overall Accuracy of classifying new fruit samples into domains: {zsl_domain_accuracy:.4f}")
    plot_confusion_matrix_heatmap(y_true_d_new, pred_domains_new,
                                  class_names=all_domain_names_list,
                                  title=f'ZSL Domain CM ({config.PHASE3_NEW_FRUIT_NAME} Samples vs All Domain Prototypes)',
                                  save_path=config.P3_ZSL_DOMAIN_CONFMAT_PLOT_PATH)
    # Plot t-SNE colored by PREDICTED domain
    plot_tsne(new_fruit_features.cpu().numpy(), pred_domains_new, domain_names_all_map_reverse,
              title=f't-SNE of {config.PHASE3_NEW_FRUIT_NAME} Features (Colored by Predicted Domain)',
              save_path=config.P3_ZSL_DOMAIN_TSNE_PLOT_PATH)


    # --- 7. Step 2: Predict Maturity using Sub-Prototypes based on Predicted Domain ---
    print(f"\nStep 2: Predicting Maturity for '{config.PHASE3_NEW_FRUIT_NAME}' using Sub-Prototypes...")
    pred_maturity_new = np.zeros_like(y_true_m_new) # Initialize predictions
    num_sub_prototypes_total = sub_prototypes.shape[0]

    with torch.no_grad():
        # Normalize features and sub-prototypes once
        query_norm = F.normalize(new_fruit_features, p=2, dim=1)
        sub_proto_norm = F.normalize(sub_prototypes, p=2, dim=1)
        sub_proto_norm = torch.nan_to_num(sub_proto_norm)

        for i in range(len(new_fruit_features)):
            feature_norm = query_norm[i:i+1] # Normalized feature for sample i
            predicted_domain_idx = pred_domains_new[i] # Domain predicted in Step 1

            if predicted_domain_idx in known_domain_reverse_map: # Predicted as Grape or Strawberry
                # Find indices of sub-prototypes belonging to this PREDICTED known domain
                relevant_subproto_indices = [
                    idx for idx, info in sub_prototype_mapping.items()
                    if info['domain_idx'] == predicted_domain_idx
                ]

                if not relevant_subproto_indices:
                    print(f"Warning: No sub-prototypes found for predicted known domain {predicted_domain_idx}. Defaulting maturity.")
                    pred_maturity_new[i] = 0 # Default to Ripe
                    continue

                # Select the relevant sub-prototype vectors (normalized)
                relevant_subprotos_norm = sub_proto_norm[relevant_subproto_indices] # (2, embedding_dim)

                # Calculate similarity to ONLY these relevant sub-prototypes
                similarities_mat = torch.mm(feature_norm, relevant_subprotos_norm.t()) # (1, 2)

                # Find the index WITHIN the relevant subset (0 or 1)
                best_relative_idx = torch.argmax(similarities_mat, dim=1).item()

                # Get the absolute index of the winning sub-prototype
                winning_subproto_idx = relevant_subproto_indices[best_relative_idx]

                # Get the maturity label from the mapping
                pred_maturity_new[i] = sub_prototype_mapping[winning_subproto_idx]['maturity_idx']

            elif predicted_domain_idx == new_domain_idx: # Predicted as Tomato (the ZSL domain)
                # Fallback: Compare against ALL known sub-prototypes
                similarities_all_sub = torch.mm(feature_norm, sub_proto_norm.t()) # (1, N_subprotos)
                best_overall_subproto_idx = torch.argmax(similarities_all_sub, dim=1).item()

                # Get maturity from the globally best matching sub-prototype
                if best_overall_subproto_idx in sub_prototype_mapping:
                     pred_maturity_new[i] = sub_prototype_mapping[best_overall_subproto_idx]['maturity_idx']
                else:
                    print(f"Warning: Globally best sub-prototype index {best_overall_subproto_idx} not in mapping. Defaulting maturity.")
                    pred_maturity_new[i] = 0 # Default

            else: # Should not happen if domain indices are contiguous
                print(f"Warning: Unexpected predicted domain index {predicted_domain_idx}. Defaulting maturity.")
                pred_maturity_new[i] = 0


    # --- Evaluate Step 2 (Maturity Classification) ---
    print(f"\n--- Maturity Classification Results for '{config.PHASE3_NEW_FRUIT_NAME}' (Two-Step Sub-Proto) ---")
    report_maturity = classification_report(y_true_m_new, pred_maturity_new,
                                target_names=["Ripe", "Rotten"], zero_division=0)
    print(report_maturity)
    maturity_accuracy_new = accuracy_score(y_true_m_new, pred_maturity_new)
    print(f"Overall Maturity Accuracy (Two-Step Sub-Proto Method): {maturity_accuracy_new:.4f}")

    plot_confusion_matrix_heatmap(y_true_m_new, pred_maturity_new,
                                  class_names=["Ripe", "Rotten"],
                                  title=f'ZSL Maturity CM ({config.PHASE3_NEW_FRUIT_NAME} - Two-Step Sub-Proto)',
                                  save_path=config.P3_MATURITY_SUBPROTO_CONFMAT_PLOT_PATH)

    inference_time = time.time() - start_time_p3
    print(f"\n--- ZSL Inference (Two-Step Sub-Proto) Completed in {inference_time:.2f} seconds ---")


if __name__ == "__main__":
    run_zsl_inference_two_step_subproto()