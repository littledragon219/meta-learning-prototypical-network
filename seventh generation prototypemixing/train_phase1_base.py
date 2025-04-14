# train_phase1_base.py
import os
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, Subset, ConcatDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score # Added imports
import joblib
import time

# Import from local modules
import config # Load config
from datapreprocessing import collect_data_from_folders, calculate_expected_length # Added calc_expected_length
from model import FeatureExtractor1DCNN, HandcraftedFeatureExtractor
from utils import (
    EarlyStopping, calculate_prototypes, prototypical_loss,
    plot_training_history, plot_tsne, plot_confusion_matrix_heatmap # Added plotting utils
)

# <<< NEW: Helper function for combined labels >>>
def get_combined_label(domain_idx, maturity_idx, num_maturity_classes):
    """Combines domain and maturity index into a single label."""
    return domain_idx * num_maturity_classes + maturity_idx

def get_subprototype_mapping(domain_map, maturity_map={"ripe": 0, "rotten": 1}):
    """Creates a mapping from combined index to domain-maturity name and maturity label."""
    subproto_map = {}
    num_maturity = len(maturity_map)
    mat_idx_to_name = {v: k for k, v in maturity_map.items()}
    for domain_name, domain_idx in domain_map.items():
        for mat_name, mat_idx in maturity_map.items():
            combined_idx = get_combined_label(domain_idx, mat_idx, num_maturity)
            subproto_map[combined_idx] = {
                "name": f"{domain_name}-{mat_name}",
                "domain_idx": domain_idx,
                "maturity_idx": mat_idx,
                "maturity_name": mat_name
            }
    return subproto_map
# <<< END NEW >>>

def train_phase1():
    print("--- Starting Training Phase 1: Base Model (Grape + Strawberry) ---")
    start_time_p1 = time.time()
    os.makedirs(config.BASE_SAVE_DIR, exist_ok=True)

    # Set Seed
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    # No GPU specific seeding needed if device is CPU

    # --- 1. Data Loading and Preparation (Phase 1 Data: Grape + Strawberry) ---
    print("Loading Phase 1 data...")
    all_h_feat_p1 = []
    all_signals_p1 = []
    all_mat_labels_p1 = [] # Keep maturity labels separate for sub-prototype calculation
    all_dom_labels_p1 = []
    domain_mapping_p1 = {}
    domain_idx_counter = 0

    # Calculate expected length once if needed
    expected_len = None
    if config.FEATURE_TYPE == 'cnn':
        expected_len = calculate_expected_length(
            config.SIGNAL_START_TIME, config.SIGNAL_END_TIME, config.SAMPLING_FREQUENCY
        )
        # Removed warning check for simplicity, assuming calculation is correct

    for fruit_name, paths in config.PHASE1_DATA_PATHS.items():
        if fruit_name not in domain_mapping_p1:
            domain_mapping_p1[fruit_name] = domain_idx_counter
            domain_idx_counter += 1
        current_domain_idx = domain_mapping_p1[fruit_name]

        h_feat, signals, mat_labels, dom_labels = collect_data_from_folders(
            fruit_name=fruit_name,
            ripe_folder=paths["ripe"],
            rotten_folder=paths["rotten"],
            domain_label=current_domain_idx, # Use mapped index
            feature_type=config.FEATURE_TYPE,
            expected_length=expected_len,
            cfg=config # Pass the config object
        )

        if h_feat.size > 0:
            all_h_feat_p1.append(h_feat)
            all_signals_p1.append(signals)
            all_mat_labels_p1.append(mat_labels) # Store maturity labels
            all_dom_labels_p1.append(dom_labels)
        else:
            print(f"Warning: No data loaded for {fruit_name} in Phase 1.")

    if not all_h_feat_p1:
        raise ValueError("No data collected for any domain in Phase 1. Check config paths.")

    # Concatenate data from all loaded domains
    h_feat_all = np.concatenate(all_h_feat_p1, axis=0)
    mat_labels_all = np.concatenate(all_mat_labels_p1, axis=0) # Concatenate maturity labels
    dom_labels_all = np.concatenate(all_dom_labels_p1, axis=0)
    if config.FEATURE_TYPE == 'cnn':
        # Filter out potential empty arrays before concatenating signals
        valid_signals = [s for s in all_signals_p1 if s is not None and s.size > 0]
        if valid_signals:
             signals_all = np.concatenate(valid_signals, axis=0)
             # Safety check on signal shape
             if signals_all.shape[0] != h_feat_all.shape[0]:
                 raise ValueError("Signal and label count mismatch after concatenation.")
        else:
             raise ValueError("CNN mode selected, but no valid signal data collected for Phase 1.")
    else:
        signals_all = np.array([]) # No signals needed for handcrafted

    num_domains_phase1 = len(domain_mapping_p1)
    domain_reverse_mapping_p1 = {v: k for k, v in domain_mapping_p1.items()}
    print(f"Loaded data for {num_domains_phase1} domains in Phase 1: {list(domain_mapping_p1.keys())}")
    print(f"Total samples loaded: {len(dom_labels_all)}")

    # Split into Train and Validation (stratify by domain)
    indices = np.arange(len(dom_labels_all))
    train_indices, val_indices = train_test_split(
        indices,
        test_size=config.PHASE1_VALIDATION_SPLIT,
        stratify=dom_labels_all, # Stratify by DOMAIN labels for consistent splits
        random_state=config.SEED
    )

    # Scale Handcrafted Features (if used) - Fit on training data only
    scaler = None
    if config.FEATURE_TYPE == 'handcrafted':
        scaler = StandardScaler()
        X_train_h_scaled = scaler.fit_transform(h_feat_all[train_indices])
        X_val_h_scaled = scaler.transform(h_feat_all[val_indices])
        joblib.dump(scaler, config.P1_SCALER_PATH)
        print(f"Scaler saved to {config.P1_SCALER_PATH}")
    else:
        X_train_h_scaled = h_feat_all[train_indices] # Not used by CNN
        X_val_h_scaled = h_feat_all[val_indices]

    # Prepare tensors based on feature type
    if config.FEATURE_TYPE == 'cnn':
        X_train = torch.tensor(signals_all[train_indices], dtype=torch.float32)
        X_val = torch.tensor(signals_all[val_indices], dtype=torch.float32)
    else: # handcrafted
        X_train = torch.tensor(X_train_h_scaled, dtype=torch.float32)
        X_val = torch.tensor(X_val_h_scaled, dtype=torch.float32)

    # Use DOMAIN labels for Phase 1 training loss (ProtoNet for domain discrimination)
    y_train_d = torch.tensor(dom_labels_all[train_indices], dtype=torch.long)
    y_val_d = torch.tensor(dom_labels_all[val_indices], dtype=torch.long)

    # <<< NEW: Keep maturity labels for sub-prototype calculation >>>
    y_train_m = torch.tensor(mat_labels_all[train_indices], dtype=torch.long)
    # y_val_m = torch.tensor(mat_labels_all[val_indices], dtype=torch.long) # Not needed for validation loop

    # <<< NEW: Create combined labels for the *training set only* for sub-prototype calculation >>>
    y_train_combined = get_combined_label(y_train_d, y_train_m, config.NUM_MATURITY_CLASSES)
    num_subproto_classes = num_domains_phase1 * config.NUM_MATURITY_CLASSES
    subprototype_mapping = get_subprototype_mapping(domain_mapping_p1)
    print(f"Sub-prototype mapping created: {subprototype_mapping}")
    print(f"Total sub-prototype classes: {num_subproto_classes}")


    # Training dataset uses domain labels for loss calculation
    train_dataset_for_loss = TensorDataset(X_train, y_train_d)
    # Validation dataset uses domain labels for validation metrics
    val_dataset = TensorDataset(X_val, y_val_d)
    # <<< NEW: Create a training dataset including combined labels for prototype calculation >>>
    train_dataset_for_proto = TensorDataset(X_train, y_train_combined) # Dataset with combined labels

    # DataLoaders
    train_loader = DataLoader(train_dataset_for_loss, batch_size=config.PHASE1_BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=config.PHASE1_BATCH_SIZE * 2, shuffle=False, num_workers=0, pin_memory=False)
    # <<< NEW: Loader for calculating prototypes on full training set >>>
    train_proto_loader = DataLoader(train_dataset_for_proto, batch_size=config.PHASE1_BATCH_SIZE * 2, shuffle=False, num_workers=0, pin_memory=False)


    print(f"Train samples: {len(train_dataset_for_loss)}, Validation samples: {len(val_dataset)}")

    # --- 2. Model Initialization ---
    if config.FEATURE_TYPE == 'cnn':
        model = FeatureExtractor1DCNN(
            sequence_length=expected_len, # Use calculated length
            embedding_dim=config.EMBEDDING_DIM
        ).to(config.DEVICE)
    else: # handcrafted
        model = HandcraftedFeatureExtractor(
            input_dim=config.HANDCRAFTED_DIM,
            embedding_dim=config.EMBEDDING_DIM
        ).to(config.DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=config.PHASE1_LR, weight_decay=config.PHASE1_WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=config.PHASE1_PATIENCE // 2, factor=0.2, verbose=True)
    early_stopping = EarlyStopping(patience=config.PHASE1_PATIENCE, verbose=True, path=config.P1_MODEL_SAVE_PATH)

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    # --- 3. Training Loop ---
    print(f"\nStarting Phase 1 training loop ({num_domains_phase1} domains)...")
    for epoch in range(config.PHASE1_EPOCHS):
        print(f"\nPhase 1 Epoch {epoch+1}/{config.PHASE1_EPOCHS}")
        model.train()
        running_loss = 0.0
        running_acc = 0.0
        num_batches = len(train_loader)

        # Calculate DOMAIN prototypes from training data at the start of epoch for loss stability
        with torch.no_grad():
            all_train_feat_epoch = []
            all_train_lbl_epoch = [] # Domain labels
            temp_train_loader_dom = DataLoader(train_dataset_for_loss, batch_size=config.PHASE1_BATCH_SIZE * 2, shuffle=False, num_workers=0)
            for inputs, domain_labels in temp_train_loader_dom:
                 inputs = inputs.to(config.DEVICE)
                 features = model(inputs)
                 all_train_feat_epoch.append(features.cpu()) # Collect features on CPU
                 all_train_lbl_epoch.append(domain_labels.cpu()) # Collect domain labels
            all_train_feat_epoch = torch.cat(all_train_feat_epoch).to(config.DEVICE) # Move features back to device
            all_train_lbl_epoch = torch.cat(all_train_lbl_epoch).to(config.DEVICE) # Move labels back
            stable_domain_prototypes_epoch = calculate_prototypes(all_train_feat_epoch, all_train_lbl_epoch, num_domains_phase1) # Calculate domain prototypes
            del temp_train_loader_dom, all_train_feat_epoch, all_train_lbl_epoch # Free memory


        for i, (inputs, domain_labels) in enumerate(train_loader): # Use loader with DOMAIN labels
            inputs, domain_labels = inputs.to(config.DEVICE), domain_labels.to(config.DEVICE)
            optimizer.zero_grad()
            query_features = model(inputs)

            # Use stable DOMAIN prototypes calculated once per epoch for the loss
            loss, acc = prototypical_loss(query_features, stable_domain_prototypes_epoch, domain_labels, distance='cosine', temperature=10.0)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_acc += acc.item() # Accuracy of classifying query points to correct domain prototype

            if (i + 1) % max(1, num_batches // 4) == 0:
                 print(f"  Batch {i+1}/{num_batches} | Loss: {loss.item():.4f} | Domain Acc: {acc.item():.4f}")


        epoch_train_loss = running_loss / num_batches
        epoch_train_acc = running_acc / num_batches
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)

        # --- Validation Loop (Uses Domain Labels) ---
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        all_val_labels_epoch = []
        all_val_preds_epoch = []
        all_val_features_epoch = [] # For t-SNE

        with torch.no_grad():
             # Use the same stable DOMAIN prototypes calculated from training data for validation
            stable_prototypes_val = stable_domain_prototypes_epoch

            for inputs, domain_labels in val_loader: # Use validation loader (domain labels)
                inputs, domain_labels = inputs.to(config.DEVICE), domain_labels.to(config.DEVICE)
                query_features = model(inputs)
                loss, acc, preds = prototypical_loss(query_features, stable_prototypes_val, domain_labels, distance='cosine', temperature=10.0, return_preds=True)
                val_loss += loss.item()
                val_acc += acc.item() # Accuracy based on domain proto assignment

                all_val_labels_epoch.extend(domain_labels.cpu().numpy())
                all_val_preds_epoch.extend(preds.cpu().numpy())
                all_val_features_epoch.append(query_features.cpu())


        epoch_val_loss = val_loss / len(val_loader)
        # epoch_val_acc = val_acc / len(val_loader) # Average proto accuracy across batches (less intuitive)
        # Calculate overall validation accuracy using sklearn (more reliable metric)
        epoch_val_acc_sk = accuracy_score(all_val_labels_epoch, all_val_preds_epoch)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc_sk) # Store sklearn accuracy

        print(f"Epoch {epoch+1} Summary: Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.4f} | Val Loss: {epoch_val_loss:.4f} Acc(SK): {epoch_val_acc_sk:.4f}")

        scheduler.step(epoch_val_loss)
        early_stopping(epoch_val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered for Phase 1.")
            break

    # --- 4. Save Artifacts & Evaluate Final Model ---
    print("\nLoading best Phase 1 model weights...")
    if os.path.exists(config.P1_MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(config.P1_MODEL_SAVE_PATH, map_location=config.DEVICE))
        print(f"Best model loaded from {config.P1_MODEL_SAVE_PATH}")
    else:
        print("Warning: Best model checkpoint not found. Using model from last epoch.")

    # --- Calculate and Save Final DOMAIN Prototypes (as before) ---
    print("Calculating and saving final Phase 1 DOMAIN prototypes from training data...")
    model.eval()
    with torch.no_grad():
         # Use the loader with domain labels
         all_train_feat_final_dom = []
         all_train_lbl_final_dom = []
         temp_train_loader_dom = DataLoader(train_dataset_for_loss, batch_size=config.PHASE1_BATCH_SIZE * 2, shuffle=False, num_workers=0)
         for inputs, domain_labels in temp_train_loader_dom:
             inputs = inputs.to(config.DEVICE)
             all_train_feat_final_dom.append(model(inputs).cpu()) # Collect features on CPU
             all_train_lbl_final_dom.append(domain_labels.cpu()) # Collect domain labels
         all_train_feat_final_dom = torch.cat(all_train_feat_final_dom).to(config.DEVICE) # Move back to device for proto calc
         all_train_lbl_final_dom = torch.cat(all_train_lbl_final_dom).to(config.DEVICE)
         final_domain_prototypes = calculate_prototypes(all_train_feat_final_dom, all_train_lbl_final_dom, num_domains_phase1)
         del temp_train_loader_dom, all_train_feat_final_dom, all_train_lbl_final_dom

    torch.save(final_domain_prototypes, config.P1_PROTOTYPES_SAVE_PATH)
    print(f"Final DOMAIN prototypes saved to {config.P1_PROTOTYPES_SAVE_PATH}")

    # <<< NEW: Calculate and Save Final SUB-PROTOTYPES >>>
    print("\nCalculating and saving final Phase 1 SUB-PROTOTYPES from training data...")
    model.eval()
    with torch.no_grad():
        all_train_feat_final_sub = []
        all_train_lbl_final_sub = [] # Combined labels
        # Use the loader with combined labels
        for inputs, combined_labels in train_proto_loader:
            inputs = inputs.to(config.DEVICE)
            all_train_feat_final_sub.append(model(inputs).cpu()) # Collect features on CPU
            all_train_lbl_final_sub.append(combined_labels.cpu()) # Collect combined labels
        all_train_feat_final_sub = torch.cat(all_train_feat_final_sub).to(config.DEVICE) # Move features back
        all_train_lbl_final_sub = torch.cat(all_train_lbl_final_sub).to(config.DEVICE) # Move combined labels back

        final_subprototypes = calculate_prototypes(
            all_train_feat_final_sub,
            all_train_lbl_final_sub,
            num_subproto_classes # Use the total number of sub-classes
        )
        del all_train_feat_final_sub, all_train_lbl_final_sub # Free memory

    # Save sub-prototypes AND the mapping
    subprototype_data_to_save = {
        'prototypes': final_subprototypes.cpu(), # Save prototypes on CPU
        'mapping': subprototype_mapping
    }
    torch.save(subprototype_data_to_save, config.P1_SUBPROTOTYPES_SAVE_PATH)
    print(f"Final SUB-PROTOTYPES and mapping saved to {config.P1_SUBPROTOTYPES_SAVE_PATH}")
    # <<< END NEW >>>


    # --- Final Validation Evaluation (Domain Classification - same as before) ---
    print("\n--- Final Phase 1 Validation Results (Best Model - Domain Classification) ---")
    model.eval()
    final_val_loss = 0.0
    final_val_labels = []
    final_val_preds = []
    final_val_features = []
    with torch.no_grad():
        for inputs, domain_labels in val_loader: # Use validation loader with domain labels
            inputs, domain_labels = inputs.to(config.DEVICE), domain_labels.to(config.DEVICE)
            query_features = model(inputs)
            # Use final calculated DOMAIN prototypes for validation metric
            loss, _, preds = prototypical_loss(query_features, final_domain_prototypes, domain_labels, distance='cosine', temperature=10.0, return_preds=True)
            final_val_loss += loss.item()
            final_val_labels.extend(domain_labels.cpu().numpy())
            final_val_preds.extend(preds.cpu().numpy())
            final_val_features.append(query_features.cpu()) # Collect features on CPU

    final_val_loss /= len(val_loader)
    final_val_features = torch.cat(final_val_features).numpy()

    print(f"Final Validation Loss (Domain): {final_val_loss:.4f}")
    report = classification_report(final_val_labels, final_val_preds,
                                   labels=list(domain_reverse_mapping_p1.keys()),
                                   target_names=[domain_reverse_mapping_p1.get(i, f"Unknown_{i}") for i in sorted(domain_reverse_mapping_p1.keys())],
                                   zero_division=0)
    print("Final Validation Classification Report (Domains):")
    print(report)
    final_accuracy = accuracy_score(final_val_labels, final_val_preds)
    print(f"Final Validation Accuracy (Domains): {final_accuracy:.4f}")

    # Plotting (remains the same, uses domain labels for coloring/CM)
    plot_training_history(history, title='Phase 1 Base Model Training', save_path=config.P1_HISTORY_PLOT_PATH)

    plot_confusion_matrix_heatmap(final_val_labels, final_val_preds,
                                  class_names=[domain_reverse_mapping_p1.get(i, f"Unknown_{i}") for i in sorted(domain_reverse_mapping_p1.keys())],
                                  title='Phase 1 Validation Confusion Matrix (Domains)',
                                  save_path=config.P1_CONFMAT_PLOT_PATH)

    plot_tsne(final_val_features, final_val_labels, domain_reverse_mapping_p1, # Use domain labels for t-SNE coloring
              title='Phase 1 Validation Feature t-SNE (Best Model)',
              save_path=config.P1_TSNE_PLOT_PATH)


    phase1_time = time.time() - start_time_p1
    print(f"--- Training Phase 1 Completed in {phase1_time:.2f} seconds ---")

if __name__ == "__main__":
    # Ensure utils.py prototypical_loss can return_preds
    train_phase1()