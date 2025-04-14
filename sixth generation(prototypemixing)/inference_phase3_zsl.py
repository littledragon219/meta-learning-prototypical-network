# inference_phase3_zsl.py
import os
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, accuracy_score # Added accuracy_score
import joblib
import time

# Import from local modules
import config # Load config
from datapreprocessing import collect_data_from_folders, calculate_expected_length # Added calc_expected_length
from model import (
    FeatureExtractor1DCNN, HandcraftedFeatureExtractor, MaturityClassifier
)
from utils import (
     plot_confusion_matrix_heatmap, plot_tsne, calculate_prototypes # Reusing calculate_prototypes
)

def load_inference_artifacts(cfg):
    """Loads artifacts needed for Phase 3 inference."""
    print("--- Loading Artifacts for Phase 3 Inference ---")
    # Scaler (needed if using handcrafted features)
    scaler = None
    if cfg.FEATURE_TYPE == 'handcrafted':
        try:
            # Load the scaler saved from Phase 1
            scaler = joblib.load(cfg.P1_SCALER_PATH)
            print(f"Loaded scaler from {cfg.P1_SCALER_PATH}")
        except FileNotFoundError:
            print(f"Error loading scaler from {cfg.P1_SCALER_PATH}. Handcrafted features cannot be scaled for the new fruit.")
            raise # Cannot proceed without scaler for handcrafted
        except Exception as e:
            print(f"Error loading scaler: {e}.")
            raise

    # Phase 2 Feature Extractor (best DG model trained on Grape + Strawberry)
    print("Loading Phase 2 feature extractor...")
    # Calculate expected length needed for CNN model init
    expected_len_cnn = None
    if cfg.FEATURE_TYPE == 'cnn':
        expected_len_cnn = calculate_expected_length(
            cfg.SIGNAL_START_TIME, cfg.SIGNAL_END_TIME, cfg.SAMPLING_FREQUENCY
        )

    if cfg.FEATURE_TYPE == 'cnn':
        model_s1 = FeatureExtractor1DCNN(
            sequence_length=expected_len_cnn, # Use calculated length
            embedding_dim=cfg.EMBEDDING_DIM
        ).to(cfg.DEVICE)
    else: # handcrafted
        model_s1 = HandcraftedFeatureExtractor(
            input_dim=cfg.HANDCRAFTED_DIM,
            embedding_dim=cfg.EMBEDDING_DIM
        ).to(cfg.DEVICE)
    try:
        # Load the *Phase 2* best model
        model_s1.load_state_dict(torch.load(cfg.P2_MODEL_SAVE_PATH, map_location=cfg.DEVICE))
        model_s1.eval()
        print(f"Loaded Phase 2 model from {cfg.P2_MODEL_SAVE_PATH}")
    except FileNotFoundError:
         print(f"Error: Phase 2 model not found at {cfg.P2_MODEL_SAVE_PATH}")
         raise
    except Exception as e:
        print(f"Error loading Phase 2 model: {e}")
        raise

    # Phase 2 Prototypes (for known domains: Grape, Strawberry)
    print("Loading Phase 2 known domain prototypes...")
    try:
        known_prototypes = torch.load(cfg.P2_PROTOTYPES_SAVE_PATH, map_location=cfg.DEVICE)
        # Infer known domains from the loaded prototypes and Phase 2 config
        known_domain_mapping_p2 = {name: i for i, name in enumerate(config.PHASE2_DATA_PATHS.keys())}
        num_known_domains = len(known_domain_mapping_p2)
        if known_prototypes.shape[0] != num_known_domains:
             print(f"Warning: Number of loaded prototypes ({known_prototypes.shape[0]}) doesn't match number of Phase 2 domains ({num_known_domains}).")
             # Adjust num_known_domains based on loaded prototypes? Or error?
             # Let's trust the loaded file for now.
             num_known_domains = known_prototypes.shape[0]

        print(f"Loaded {num_known_domains} known prototypes from {cfg.P2_PROTOTYPES_SAVE_PATH}")
        known_domain_reverse_mapping_p2 = {v: k for k, v in known_domain_mapping_p2.items() if v < num_known_domains} # Ensure mapping matches loaded prototypes count

    except FileNotFoundError:
        print(f"Error: Phase 2 prototypes not found at {cfg.P2_PROTOTYPES_SAVE_PATH}")
        raise
    except Exception as e:
        print(f"Error loading Phase 2 prototypes: {e}")
        raise

    # Stage 2 Maturity Classifiers (for known domains: Grape, Strawberry)
    print("Loading Stage 2 maturity classifiers...")
    stage2_maturity_models = {}
    # Load models for domains present in known_domain_reverse_mapping_p2
    for domain_idx, domain_name in known_domain_reverse_mapping_p2.items():
        s2_model_path = cfg.S2_MODEL_SAVE_TEMPLATE.format(domain_name)
        if os.path.exists(s2_model_path):
            try:
                model_s2 = MaturityClassifier(
                    input_dim=cfg.EMBEDDING_DIM, num_classes=2
                ).to(cfg.DEVICE)
                model_s2.load_state_dict(torch.load(s2_model_path, map_location=cfg.DEVICE))
                model_s2.eval()
                stage2_maturity_models[domain_idx] = model_s2 # Store model keyed by domain index
                print(f"  Loaded S2 model for {domain_name} (Index {domain_idx}) from {s2_model_path}")
            except Exception as e:
                print(f"  Error loading S2 model for {domain_name}: {e}")
        else:
             print(f"  Warning: S2 model for {domain_name} not found at {s2_model_path}.")

    # Return the mapping for KNOWN domains only
    return scaler, model_s1, known_prototypes, stage2_maturity_models, known_domain_reverse_mapping_p2


def initialize_zsl_prototype(new_fruit_features, known_prototypes, k=1, mode='average'):
    """
    Initializes a prototype for a new (ZSL) class by combining k nearest known prototypes.

    Args:
        new_fruit_features (Tensor): Features of the new fruit samples (N_new, embedding_dim).
        known_prototypes (Tensor): Prototypes of known classes (N_known, embedding_dim).
        k (int): Number of nearest neighbors to consider.
        mode (str): 'average' or 'weighted_average'.

    Returns:
        Tensor: The initialized prototype for the new fruit (1, embedding_dim).
    """
    if new_fruit_features.numel() == 0:
         print("Warning: Cannot initialize ZSL prototype with no features.")
         # Return zeros matching embedding dim
         return torch.zeros(1, known_prototypes.shape[1], device=known_prototypes.device)

    # Calculate average feature vector for the new fruit samples
    avg_new_feature = new_fruit_features.mean(dim=0, keepdim=True) # (1, embedding_dim)

    # Ensure avg_new_feature and known_prototypes are on the same device
    avg_new_feature = avg_new_feature.to(known_prototypes.device)


    # Calculate distances (Euclidean might be more intuitive here than cosine for 'nearest')
    dists = torch.cdist(avg_new_feature, known_prototypes).squeeze(0) # (N_known,)


    # Find k nearest known prototypes
    k = min(k, len(known_prototypes)) # Ensure k is not larger than available prototypes
    if k == 0:
        print("Warning: k=0 requested for ZSL init, cannot select prototypes. Returning zero prototype.")
        return torch.zeros(1, known_prototypes.shape[1], device=known_prototypes.device)

    nearest_indices = torch.argsort(dists)[:k]
    nearest_dists = dists[nearest_indices]

    # Combine the k nearest prototypes
    selected_prototypes = known_prototypes[nearest_indices] # (k, embedding_dim)

    print(f"  Initializing ZSL prototype using {k} nearest known prototypes (Indices: {nearest_indices.cpu().numpy()}, Dists: {nearest_dists.cpu().numpy()}).")


    if mode == 'average':
        initialized_prototype = selected_prototypes.mean(dim=0, keepdim=True)
    elif mode == 'weighted_average':
        # Use inverse distance as weights (add epsilon to avoid division by zero)
        weights = 1.0 / (nearest_dists + 1e-6)
        weights = weights / weights.sum() # Normalize weights
        weights = weights.unsqueeze(1) # Make weights broadcast correctly (k, 1)
        initialized_prototype = torch.sum(selected_prototypes * weights, dim=0, keepdim=True)
    else:
        raise ValueError(f"Invalid ZSL initialization mode: {mode}")

    return initialized_prototype


def run_zsl_inference():
    # --- Phase 3: Test on Tomato ---
    print(f"--- Starting Phase 3: Zero-Shot Inference on '{config.PHASE3_NEW_FRUIT_NAME}' ---")
    start_time_p3 = time.time()
    os.makedirs(config.BASE_SAVE_DIR, exist_ok=True) # Ensure save dir exists


    # 1. Load Artifacts (P2 Model, P2 Known Prototypes, P1 Scaler, S2 Classifiers)
    try:
        scaler, model_s1, known_prototypes, stage2_maturity_models, \
            known_domain_reverse_mapping = load_inference_artifacts(config)
    except Exception as e:
        print(f"Failed to load necessary artifacts for Phase 3. Aborting. Error: {e}")
        return

    num_known_domains = known_prototypes.shape[0]
    # Assign the next available index to the new domain (Tomato)
    new_domain_idx = num_known_domains

    # Create a full domain mapping including the new fruit for reporting
    domain_names_all_map = {**known_domain_reverse_mapping, new_domain_idx: config.PHASE3_NEW_FRUIT_NAME}
    print(f"\nDomain mapping for inference: {domain_names_all_map}")


    # --- 2. Load and Prepare New Fruit Data (Tomato) ---
    print(f"\nLoading data for new fruit: '{config.PHASE3_NEW_FRUIT_NAME}'")
    # Calculate expected length once if needed
    expected_len = None
    if config.FEATURE_TYPE == 'cnn':
        expected_len = calculate_expected_length(
            config.SIGNAL_START_TIME, config.SIGNAL_END_TIME, config.SAMPLING_FREQUENCY
        )

    h_feat_new, signals_new, mat_labels_new, _ = collect_data_from_folders(
        fruit_name=config.PHASE3_NEW_FRUIT_NAME,
        ripe_folder=config.PHASE3_NEW_FRUIT_PATHS["ripe"],
        rotten_folder=config.PHASE3_NEW_FRUIT_PATHS["rotten"],
        domain_label=new_domain_idx, # Assign the new domain index (important!)
        feature_type=config.FEATURE_TYPE,
        expected_length=expected_len,
        cfg=config, # Pass config
        n_augments=0 # No augmentation during inference
    )

    if h_feat_new.size == 0:
        print(f"Error: No data found for the new fruit '{config.PHASE3_NEW_FRUIT_NAME}' at specified paths. Cannot perform inference.")
        return

    print(f"Loaded {len(mat_labels_new)} samples for '{config.PHASE3_NEW_FRUIT_NAME}'.")

    # Prepare input tensor for the new fruit (Tomato)
    if config.FEATURE_TYPE == 'cnn':
        if signals_new.size == 0:
             print("Error: CNN mode selected but no signal data loaded for the new fruit.")
             return
        X_new_tensor = torch.tensor(signals_new, dtype=torch.float32).to(config.DEVICE) #.unsqueeze(1) handled in model
    else: # handcrafted
        if scaler is None:
             print("Error: Scaler not loaded, cannot process handcrafted features for new fruit.")
             return
        try:
            h_feat_new_scaled = scaler.transform(h_feat_new)
        except Exception as e:
            print(f"Error scaling handcrafted features for new fruit: {e}")
            return
        X_new_tensor = torch.tensor(h_feat_new_scaled, dtype=torch.float32).to(config.DEVICE)

    # True labels for the new fruit samples
    y_true_m_new = torch.tensor(mat_labels_new, dtype=torch.long)
    # True domain label for all these samples is the new index
    y_true_d_new = torch.full((len(y_true_m_new),), new_domain_idx, dtype=torch.long)


    # --- 3. Extract Features for New Fruit (Tomato) ---
    print("Extracting features for the new fruit...")
    model_s1.eval() # Ensure model is in eval mode
    with torch.no_grad():
        new_fruit_features = model_s1(X_new_tensor) # (N_new, embedding_dim)
    print(f"Extracted features shape: {new_fruit_features.shape}")

    # --- 4. Initialize ZSL Prototype for Tomato ---
    print("Initializing ZSL prototype for the new fruit...")
    zsl_prototype = initialize_zsl_prototype(
        new_fruit_features, # Use features extracted from Tomato samples
        known_prototypes, # Prototypes of Grape and Strawberry
        k=config.PHASE3_ZSL_INIT_KNN,
        mode=config.PHASE3_ZSL_INIT_MODE
    )
    print(f"  Initialized ZSL prototype shape: {zsl_prototype.shape}")

    # --- 5. Combine Prototypes and Perform Domain Classification ---
    # Combine known (Grape, Strawberry) + initialized ZSL (Tomato) prototypes
    all_prototypes_for_inference = torch.cat([known_prototypes, zsl_prototype], dim=0) # (N_known + 1, embedding_dim)
    num_total_domains_inference = all_prototypes_for_inference.shape[0]
    print(f"Total prototypes for inference: {num_total_domains_inference}")

    print(f"\nPerforming ZSL Domain Classification (Classifying '{config.PHASE3_NEW_FRUIT_NAME}' against all prototypes)...")
    model_s1.eval() # Ensure model is still in eval
    with torch.no_grad():
         # Classify new fruit features against ALL prototypes (known + ZSL)
         # Use cosine distance/similarity for classification consistency with training
         # Higher similarity (lower distance) = closer match
        # logits = F.cosine_similarity(new_fruit_features.unsqueeze(1), all_prototypes_for_inference.unsqueeze(0), dim=-1)
        # Use normalized dot product (equivalent to cosine similarity)
        logits = torch.mm(F.normalize(new_fruit_features, p=2, dim=1),
                           F.normalize(all_prototypes_for_inference, p=2, dim=1).t())

        pred_domains_new = torch.argmax(logits, dim=1).cpu().numpy() # Predicted domain indices

    # --- 6. Evaluate ZSL Domain Classification ---
    print(f"\n--- ZSL Domain Classification Results for '{config.PHASE3_NEW_FRUIT_NAME}' Samples ---")
    # `y_true_d_new` contains the correct index for Tomato
    # `pred_domains_new` contains the predicted index (could be Grape, Strawberry, or Tomato index)
    # Use domain_names_all_map for target names
    report_domain = classification_report(y_true_d_new.numpy(), pred_domains_new,
                                labels=list(domain_names_all_map.keys()), # Ensure all possible predicted labels are included
                                target_names=[domain_names_all_map.get(i, f'Unknown_{i}') for i in sorted(domain_names_all_map.keys())],
                                zero_division=0)
    print(report_domain)

    zsl_domain_accuracy = accuracy_score(y_true_d_new.numpy(), pred_domains_new)
    print(f"Overall Accuracy of classifying new fruit samples into domains: {zsl_domain_accuracy:.4f}")

    # Plot Confusion Matrix for Domain Classification
    plot_confusion_matrix_heatmap(y_true_d_new.numpy(), pred_domains_new,
                                  class_names=[domain_names_all_map.get(i, f'Unknown_{i}') for i in sorted(domain_names_all_map.keys())],
                                  title=f'ZSL Domain CM ({config.PHASE3_NEW_FRUIT_NAME} Samples vs All Prototypes)',
                                  save_path=config.P3_ZSL_DOMAIN_CONFMAT_PLOT_PATH)

    # Plot t-SNE of Tomato features colored by PREDICTED domain
    plot_tsne(new_fruit_features.cpu().numpy(), pred_domains_new, domain_names_all_map,
              title=f't-SNE of {config.PHASE3_NEW_FRUIT_NAME} Features (Colored by Predicted Domain)',
              save_path=config.P3_ZSL_DOMAIN_TSNE_PLOT_PATH)


    # --- 7. (Attempt) Maturity Classification for New Fruit (Tomato) ---
    # This uses the Stage 2 head corresponding to the *predicted* domain for each sample.
    # Since no S2 model exists for Tomato, predictions where domain=Tomato index need a fallback.
    print(f"\nAttempting Maturity Classification for '{config.PHASE3_NEW_FRUIT_NAME}' samples (using S2 models of predicted domains)...")
    pred_maturity_new = np.zeros_like(y_true_m_new.numpy()) # Initialize predictions array

    fallback_predictions = 0
    successful_predictions = 0

    with torch.no_grad():
        for i in range(len(new_fruit_features)):
            predicted_domain_idx = pred_domains_new[i]
            feature_input = new_fruit_features[i:i+1] # Get feature for the i-th sample

            if predicted_domain_idx == new_domain_idx:
                # Fallback: Predicted as the ZSL class (Tomato), no S2 model exists.
                # Option 1: Predict majority class (e.g., Ripe=0)
                # Option 2: Use the S2 model of the *closest known* domain (based on prototype distance to *known* protos) - More complex
                # Let's use Option 1 (predict Ripe=0) for simplicity.
                pred_maturity_new[i] = 0 # Default prediction (Ripe)
                fallback_predictions += 1
            elif predicted_domain_idx in stage2_maturity_models:
                # Predicted as a known domain (Grape/Strawberry) for which we have an S2 model
                model_s2 = stage2_maturity_models[predicted_domain_idx]
                model_s2.eval() # Ensure S2 model is in eval mode
                maturity_logits = model_s2(feature_input)
                pred_maturity_new[i] = torch.argmax(maturity_logits, dim=1).cpu().item()
                successful_predictions += 1
            else:
                # Predicted a known domain index, but its S2 model wasn't loaded/found (shouldn't happen if load was ok)
                print(f"Warning: Predicted known domain {predicted_domain_idx}, but S2 model not found. Using fallback.")
                pred_maturity_new[i] = 0 # Default prediction (Ripe)
                fallback_predictions += 1

    print(f"  Maturity predictions: {successful_predictions} used S2 models, {fallback_predictions} used fallback.")

    # Evaluate the speculative maturity classification
    print(f"\n--- Maturity Classification Results for '{config.PHASE3_NEW_FRUIT_NAME}' Samples (Speculative) ---")
    # Compare `pred_maturity_new` with `y_true_m_new`
    report_maturity = classification_report(y_true_m_new.numpy(), pred_maturity_new,
                                target_names=["Ripe", "Rotten"], zero_division=0)
    print(report_maturity)

    maturity_accuracy_new = accuracy_score(y_true_m_new.numpy(), pred_maturity_new)
    print(f"Overall Maturity Accuracy (using predicted domain's S2 model or fallback): {maturity_accuracy_new:.4f}")

    # Plot Confusion Matrix for Maturity Classification
    plot_confusion_matrix_heatmap(y_true_m_new.numpy(), pred_maturity_new,
                                  class_names=["Ripe", "Rotten"],
                                  title=f'ZSL Maturity CM ({config.PHASE3_NEW_FRUIT_NAME} Samples - Speculative)',
                                  save_path=config.P3_ZSL_MATURITY_CONFMAT_PLOT_PATH)

    inference_time = time.time() - start_time_p3
    print(f"\n--- ZSL Inference Completed in {inference_time:.2f} seconds ---")


if __name__ == "__main__":
    run_zsl_inference()