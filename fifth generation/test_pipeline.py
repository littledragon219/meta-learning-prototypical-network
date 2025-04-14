# test_pipeline.py
import os
import numpy as np
import torch
import joblib
from sklearn.metrics import classification_report, accuracy_score
import time
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
# Import from local modules
from datapreprocessing import (
    collect_data_from_folders, calculate_expected_length,
    DEFAULT_SIGNAL_START_TIME, DEFAULT_SIGNAL_END_TIME, DEFAULT_SAMPLING_FREQUENCY,
    DEFAULT_AUGMENTATIONS
)
from model import (
    FeatureExtractor1DCNN, HandcraftedFeatureExtractor,
    ProtoNetDomainClassifier, MaturityClassifier # Need classes to instantiate models
)
from utils import (
    plot_confusion_matrix_heatmap, plot_tsne
)

def load_models(config, device):
    """Loads the trained Stage 1 and Stage 2 models."""
    # --- Load Stage 1 Feature Extractor ---
    print(f"Loading Stage 1 Feature Extractor ({config['feature_type']})...")
    if config['feature_type'] == 'cnn':
        feature_extractor_s1 = FeatureExtractor1DCNN(
            input_channels=1,
            sequence_length=config['expected_signal_length'],
            embedding_dim=config['embedding_dim']
        ).to(device)
        input_data_index = 1 # Use signal data
    elif config['feature_type'] == 'handcrafted':
         # Need input dim for handcrafted features - determine from loaded data later or assume 5
         # For now, assume 5, but ideally get from scaler or data
         num_handcrafted_features = 5 # joblib.load(config['scaler_path']).n_features_in_
         feature_extractor_s1 = HandcraftedFeatureExtractor(
            input_dim=num_handcrafted_features,
            embedding_dim=config['embedding_dim']
         ).to(device)
         input_data_index = 0 # Use handcrafted features
    else:
        raise ValueError("Invalid feature_type in config")

    try:
        feature_extractor_s1.load_state_dict(torch.load(config['s1_model_path'], map_location=device))
        feature_extractor_s1.eval()
        print(f"  Loaded from: {config['s1_model_path']}")
    except Exception as e:
        print(f"Error loading Stage 1 model: {e}")
        raise

    # --- Load Stage 1 Prototypes ---
    print("Loading Stage 1 Prototypes...")
    try:
        prototypes_s1 = torch.load(config['s1_prototypes_path'], map_location=device)
        print(f"  Loaded {prototypes_s1.shape[0]} prototypes from: {config['s1_prototypes_path']}")
        if prototypes_s1.shape[0] != config['num_domains']:
             print(f"Warning: Number of loaded prototypes ({prototypes_s1.shape[0]}) doesn't match NUM_DOMAINS ({config['num_domains']}).")
    except Exception as e:
        print(f"Error loading Stage 1 prototypes: {e}")
        raise

    # --- Load Stage 2 Maturity Classifiers ---
    print("Loading Stage 2 Maturity Classifiers...")
    stage2_models = {}
    for domain_idx, domain_name in config['domain_reverse_mapping'].items():
        model_path = config['s2_model_template'].format(domain_name)
        print(f"  Attempting to load model for {domain_name} from {model_path}...")
        if os.path.exists(model_path):
            try:
                model_s2 = MaturityClassifier(
                    input_dim=config['embedding_dim'], # Takes Stage 1 embedding as input
                    num_classes=2
                ).to(device)
                model_s2.load_state_dict(torch.load(model_path, map_location=device))
                model_s2.eval()
                stage2_models[domain_idx] = model_s2
                print(f"    Successfully loaded model for {domain_name}.")
            except Exception as e:
                print(f"    Error loading Stage 2 model for {domain_name}: {e}. Skipping.")
        else:
            print(f"    Model file not found for {domain_name}. Skipping.")

    if not stage2_models:
        print("Warning: No Stage 2 models were loaded successfully.")

    return feature_extractor_s1, prototypes_s1, stage2_models, input_data_index


def prepare_test_data(config, device):
    """Loads and prepares test data, either saved or from new folders."""

    if config.get("use_saved_test_data", False):
        # --- Load Test Data Saved During Training ---
        print(f"Loading saved test data from: {config['test_data_path']}")
        try:
            test_data = np.load(config['test_data_path'])
            X_test_h_scaled = test_data['X_test_h']
            X_test_s = test_data['X_test_s']
            y_test_m = test_data['y_test_m']
            y_test_d = test_data['y_test_d']
            print("  Successfully loaded saved test data.")
            # No need for scaler here as data is already scaled
        except Exception as e:
            print(f"Error loading saved test data: {e}")
            raise
    else:
        # --- Load New Test Data from Folders ---
        print("Loading new test data from specified folders...")
        if not config.get("new_test_data_folders"):
            raise ValueError("Configuration must provide 'new_test_data_folders' when not using saved data.")

        expected_length = calculate_expected_length(
             DEFAULT_SIGNAL_START_TIME, DEFAULT_SIGNAL_END_TIME, DEFAULT_SAMPLING_FREQUENCY
        )

        all_h_features = []
        all_signals = []
        all_mat_labels = []
        all_dom_labels = []

        for fruit_name, paths in config["new_test_data_folders"].items():
            if fruit_name not in config["domain_mapping"]:
                 print(f"Warning: Fruit '{fruit_name}' in test folders not found in domain mapping. Skipping.")
                 continue
            domain_idx = config["domain_mapping"][fruit_name]

            # Check if paths exist
            ripe_path = paths.get("ripe", None)
            rotten_path = paths.get("rotten", None)
            if not ripe_path or not os.path.exists(ripe_path):
                 print(f"Warning: Ripe path not found or doesn't exist for {fruit_name}: {ripe_path}. Skipping ripe data.")
                 ripe_path = None
            if not rotten_path or not os.path.exists(rotten_path):
                 print(f"Warning: Rotten path not found or doesn't exist for {fruit_name}: {rotten_path}. Skipping rotten data.")
                 rotten_path = None

            if not ripe_path and not rotten_path:
                 print(f"Warning: No valid ripe or rotten paths for {fruit_name}. Skipping this fruit.")
                 continue

            # Use collect_data_from_folders - it handles missing folders internally now
            h_feat, signals, mat_labels, dom_labels = collect_data_from_folders(
                fruit_name=fruit_name,
                ripe_folder=ripe_path if ripe_path else "dummy_nonexistent_ripe", # Provide dummy paths if None
                rotten_folder=rotten_path if rotten_path else "dummy_nonexistent_rotten",
                domain_label=domain_idx,
                expected_length=expected_length,
                n_augments=0 # IMPORTANT: Set n_augments to 0 for testing to avoid artificial inflation
            )
            if h_feat.size > 0:
                all_h_features.append(h_feat)
                all_signals.append(signals)
                all_mat_labels.append(mat_labels)
                all_dom_labels.append(dom_labels)

        if not all_h_features:
             raise ValueError("No new test data collected. Check 'new_test_data_folders' paths.")

        X_test_h = np.concatenate(all_h_features, axis=0)
        X_test_s = np.concatenate(all_signals, axis=0)
        y_test_m = np.concatenate(all_mat_labels, axis=0)
        y_test_d = np.concatenate(all_dom_labels, axis=0)

        # --- Scale Handcrafted Features using SAVED Scaler ---
        print(f"Loading scaler from: {config['scaler_path']}")
        try:
            scaler = joblib.load(config['scaler_path'])
            print("Applying scaler to new test data's handcrafted features...")
            X_test_h_scaled = scaler.transform(X_test_h)
        except Exception as e:
            print(f"Error loading or applying scaler: {e}")
            raise

        print(f"Loaded {len(y_test_d)} new test samples.")

    # --- Convert to Tensors ---
    tensor_h = torch.tensor(X_test_h_scaled, dtype=torch.float32)
    tensor_s = torch.tensor(X_test_s, dtype=torch.float32).unsqueeze(1) # Add channel dim
    tensor_m = torch.tensor(y_test_m, dtype=torch.long)
    tensor_d = torch.tensor(y_test_d, dtype=torch.long)

    return tensor_h, tensor_s, tensor_m, tensor_d


def evaluate_on_test_set(config, save_plots=False):
    """Performs the full two-stage evaluation on test data."""
    device = config['device']
    start_time = time.time()
    print("\n--- Starting Test Set Evaluation ---")

    # 1. Load Models and Prototypes
    try:
        model_s1, prototypes_s1, stage2_models, input_data_index = load_models(config, device)
    except Exception as e:
        print(f"Failed to load models: {e}")
        return

    # 2. Load and Prepare Test Data
    try:
        tensor_h, tensor_s, y_true_m, y_true_d = prepare_test_data(config, device)
        # Select input tensor based on feature type
        test_input_tensor = tensor_s if config['feature_type'] == 'cnn' else tensor_h
    except Exception as e:
        print(f"Failed to prepare test data: {e}")
        return

    if len(test_input_tensor) == 0:
        print("No test data available to evaluate.")
        return

    print(f"\nEvaluating on {len(y_true_d)} test samples.")

    # 3. Stage 1 Inference (Domain Prediction)
    print("Performing Stage 1 Inference (Domain Classification)...")
    all_s1_features = []
    pred_domains_s1 = []
    model_s1.eval() # Ensure eval mode
    with torch.no_grad():
        # Process in batches if dataset is large
        test_loader_s1 = DataLoader(TensorDataset(test_input_tensor), batch_size=128)
        for batch_inputs, in test_loader_s1:
            inputs = batch_inputs.to(device)
            features = model_s1(inputs) # Get embeddings (batch_size, embedding_dim)
            all_s1_features.append(features.cpu())

            # Calculate distances to prototypes
            # distances = torch.cdist(features, prototypes_s1) # Euclidean distance (batch, num_domains)
            # Cosine distance: -(cosine_similarity) because lower distance should be better
            distances = -F.cosine_similarity(features.unsqueeze(1), prototypes_s1.unsqueeze(0), dim=-1)

            batch_preds = torch.argmin(distances, dim=1)
            pred_domains_s1.append(batch_preds.cpu())

    all_s1_features = torch.cat(all_s1_features).numpy()
    pred_domains_s1 = torch.cat(pred_domains_s1).numpy()
    y_true_d_np = y_true_d.cpu().numpy() # Ensure labels are numpy

    print("\n--- Stage 1 (Domain) Results ---")
    print(classification_report(y_true_d_np, pred_domains_s1,
                                target_names=config['domain_reverse_mapping'].values(), zero_division=0))
    s1_accuracy = accuracy_score(y_true_d_np, pred_domains_s1)
    print(f"Stage 1 Accuracy: {s1_accuracy:.4f}")

    if save_plots:
         plot_confusion_matrix_heatmap(y_true_d_np, pred_domains_s1,
                                       class_names=config['domain_reverse_mapping'].values(),
                                       title='Stage 1: Domain Classification Confusion Matrix (Test Set)',
                                       save_path=os.path.join(config['base_save_dir'], 'test_confusion_matrix_stage1.png'))
         plot_tsne(all_s1_features, y_true_d_np, config['domain_reverse_mapping'],
                   title='Stage 1 Features t-SNE (Colored by True Domain)',
                   save_path=os.path.join(config['base_save_dir'], 'test_tsne_stage1_domains.png'))


    # 4. Stage 2 Inference (Maturity Prediction using Predicted Domain)
    print("\nPerforming Stage 2 Inference (Maturity Classification)...")
    y_pred_m_s2 = np.zeros_like(y_true_d_np) # Initialize maturity predictions
    y_true_m_np = y_true_m.cpu().numpy()

    # Need features from Stage 1 for Stage 2 input
    s1_features_tensor = torch.tensor(all_s1_features).to(device)

    with torch.no_grad():
        for i in range(len(s1_features_tensor)):
            predicted_domain = pred_domains_s1[i]
            if predicted_domain in stage2_models:
                model_s2 = stage2_models[predicted_domain]
                # Input to Stage 2 is the feature vector from Stage 1
                feature_input = s1_features_tensor[i:i+1] # Keep batch dim
                maturity_logits = model_s2(feature_input)
                y_pred_m_s2[i] = torch.argmax(maturity_logits, dim=1).cpu().item()
            else:
                # Handle missing Stage 2 model - predict majority class (e.g., 0=Ripe) or a special value
                y_pred_m_s2[i] = 0 # Or -1 if you want to filter later
                # print(f"Warning: No Stage 2 model for predicted domain {predicted_domain}. Defaulting prediction for sample {i}.")

    print("\n--- Stage 2 (Maturity) Overall Results (using predicted domains) ---")
    # Filter out potential -1 predictions if used
    valid_indices = y_pred_m_s2 != -1
    if np.any(valid_indices):
        print(classification_report(y_true_m_np[valid_indices], y_pred_m_s2[valid_indices],
                                    target_names=["Ripe", "Rotten"], zero_division=0))
        s2_accuracy = accuracy_score(y_true_m_np[valid_indices], y_pred_m_s2[valid_indices])
        print(f"Stage 2 Overall Accuracy: {s2_accuracy:.4f}")

        if save_plots:
             plot_confusion_matrix_heatmap(y_true_m_np[valid_indices], y_pred_m_s2[valid_indices],
                                           class_names=["Ripe", "Rotten"],
                                           title='Stage 2: Overall Maturity Confusion Matrix (Test Set)',
                                           save_path=os.path.join(config['base_save_dir'], 'test_confusion_matrix_stage2_overall.png'))
             # Plot tSNE colored by true maturity
             plot_tsne(all_s1_features[valid_indices], y_true_m_np[valid_indices], {0: "Ripe", 1: "Rotten"},
                       title='Stage 1 Features t-SNE (Colored by True Maturity)',
                       save_path=os.path.join(config['base_save_dir'], 'test_tsne_stage1_maturity.png'))
    else:
        print("No valid Stage 2 predictions were made.")


    eval_time = time.time() - start_time
    print(f"\n--- Test Set Evaluation Completed in {eval_time:.2f} seconds ---")


if __name__ == "__main__":
    # --- Configuration for Standalone Testing ---
    # Define paths to your NEW, UNSEEN test data folders here
    NEW_TEST_DATA_FOLDERS = {
        "grape": {
             "ripe": r"D:/大二下学期/CDIO/grape_ripe_test",       # <--- CHANGE TO YOUR NEW GRAPE RIPE TEST FOLDER
             "rotten": r"D:/大二下学期/CDIO/grape_rotten_test"     # <--- CHANGE TO YOUR NEW GRAPE ROTTEN TEST FOLDER
             },
        "strawberry": {
             "ripe": r"D:/大二下学期/CDIO/strawberry_ripe_test", # <--- CHANGE TO YOUR NEW STRAWBERRY RIPE TEST FOLDER
             "rotten": r"D:/大二下学期/CDIO/strawberry_rotten_test" # <--- CHANGE TO YOUR NEW STRAWBERRY ROTTEN TEST FOLDER
             },
         "tomato": {
              "ripe": r"D:/大二下学期/CDIO/tomato_ripe_test",    # <--- CHANGE TO YOUR NEW TOMATO RIPE TEST FOLDER
              "rotten": r"D:/大二下学期/CDIO/tomato_rotten_test"  # <--- CHANGE TO YOUR NEW TOMATO ROTTEN TEST FOLDER
              },
         # Add more fruits if your trained model supports them
    }

    # --- Match settings used during training ---
    TRAINED_FEATURE_TYPE = 'handcrafted' # MUST match the FEATURE_TYPE used in train_pipeline.py
    TRAINED_EMBEDDING_DIM = 64       # MUST match the EMBEDDING_DIM used for the feature type

    # --- Load Domain Mapping from Training (or redefine if needed) ---
    # This should ideally be saved during training, but we redefine for standalone use
    DOMAIN_MAPPING_TEST = {"grape": 0, "strawberry": 1, "tomato": 2} # Ensure this matches training
    DOMAIN_REVERSE_MAPPING_TEST = {v: k for k, v in DOMAIN_MAPPING_TEST.items()}
    NUM_DOMAINS_TEST = len(DOMAIN_MAPPING_TEST)

    # --- Define Paths to Saved Training Artifacts ---
    BASE_SAVE_DIR_TEST = "pipeline_models_v2" # Directory where training artifacts are saved

    test_pipeline_config = {
        "base_save_dir": BASE_SAVE_DIR_TEST,
        "feature_type": TRAINED_FEATURE_TYPE,
        "embedding_dim": TRAINED_EMBEDDING_DIM,
        "num_domains": NUM_DOMAINS_TEST,
        "domain_mapping": DOMAIN_MAPPING_TEST,
        "domain_reverse_mapping": DOMAIN_REVERSE_MAPPING_TEST,
        "expected_signal_length": calculate_expected_length(
            DEFAULT_SIGNAL_START_TIME, DEFAULT_SIGNAL_END_TIME, DEFAULT_SAMPLING_FREQUENCY
        ),
        "use_saved_test_data": False, # Set to False to load NEW data
        "test_data_path": None,       # Not used when use_saved_test_data is False
        "new_test_data_folders": NEW_TEST_DATA_FOLDERS,
        "scaler_path": os.path.join(BASE_SAVE_DIR_TEST, 'scaler.joblib'),
        "s1_model_path": os.path.join(BASE_SAVE_DIR_TEST, f'stage1_feature_extractor_{TRAINED_FEATURE_TYPE}_best.pth'),
        "s1_prototypes_path": os.path.join(BASE_SAVE_DIR_TEST, f'stage1_domain_prototypes_{TRAINED_FEATURE_TYPE}.pth'),
        "s2_model_template": os.path.join(BASE_SAVE_DIR_TEST, f'stage2_maturity_classifier_{{}}_using_{TRAINED_FEATURE_TYPE}_best.pth'),
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
    }

    evaluate_on_test_set(test_pipeline_config, save_plots=True)