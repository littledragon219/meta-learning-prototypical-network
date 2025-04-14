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
    all_mat_labels_p1 = []
    all_dom_labels_p1 = []
    domain_mapping_p1 = {}
    domain_idx_counter = 0

    # Calculate expected length once if needed
    expected_len = None
    if config.FEATURE_TYPE == 'cnn':
        expected_len = calculate_expected_length(
            config.SIGNAL_START_TIME, config.SIGNAL_END_TIME, config.SAMPLING_FREQUENCY
        )
        if expected_len != config.EXPECTED_SIGNAL_LENGTH:
             print(f"Warning: Calculated expected length {expected_len} differs from config {config.EXPECTED_SIGNAL_LENGTH}. Using calculated value.")
             # Optionally update config or raise error if consistency is critical
             # config.EXPECTED_SIGNAL_LENGTH = expected_len


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
            all_mat_labels_p1.append(mat_labels)
            all_dom_labels_p1.append(dom_labels)
        else:
            print(f"Warning: No data loaded for {fruit_name} in Phase 1.")

    if not all_h_feat_p1:
        raise ValueError("No data collected for any domain in Phase 1. Check config paths.")

    # Concatenate data from all loaded domains
    h_feat_all = np.concatenate(all_h_feat_p1, axis=0)
    mat_labels_all = np.concatenate(all_mat_labels_p1, axis=0)
    dom_labels_all = np.concatenate(all_dom_labels_p1, axis=0)
    if config.FEATURE_TYPE == 'cnn':
        # Filter out potential empty arrays before concatenating signals
        valid_signals = [s for s in all_signals_p1 if s is not None and s.size > 0]
        if valid_signals:
             signals_all = np.concatenate(valid_signals, axis=0)
             # Safety check on signal shape
             if signals_all.shape[0] != h_feat_all.shape[0]:
                 print(f"Warning: Mismatch between signal count ({signals_all.shape[0]}) and feature/label count ({h_feat_all.shape[0]})")
                 # Handle mismatch, maybe raise error or try to align
                 raise ValueError("Signal and label count mismatch after concatenation.")
        else:
             print("Warning: No valid signal data collected for CNN mode.")
             signals_all = np.array([]) # Empty array if no signals
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
        stratify=dom_labels_all, # Stratify by DOMAIN labels
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
        X_train_h_scaled = h_feat_all[train_indices] # Won't be used directly by CNN
        X_val_h_scaled = h_feat_all[val_indices]

    # Prepare tensors based on feature type
    if config.FEATURE_TYPE == 'cnn':
        if signals_all.size == 0:
             raise ValueError("CNN feature type selected, but no signal data was loaded/processed correctly.")
        X_train = torch.tensor(signals_all[train_indices], dtype=torch.float32) #.unsqueeze(1) -> handled in model
        X_val = torch.tensor(signals_all[val_indices], dtype=torch.float32) #.unsqueeze(1)
    else: # handcrafted
        X_train = torch.tensor(X_train_h_scaled, dtype=torch.float32)
        X_val = torch.tensor(X_val_h_scaled, dtype=torch.float32)

    # Use DOMAIN labels for Phase 1 training (ProtoNet for domain discrimination)
    y_train_d = torch.tensor(dom_labels_all[train_indices], dtype=torch.long)
    y_val_d = torch.tensor(dom_labels_all[val_indices], dtype=torch.long)

    train_dataset = TensorDataset(X_train, y_train_d)
    val_dataset = TensorDataset(X_val, y_val_d)

    # Use smaller num_workers for CPU
    train_loader = DataLoader(train_dataset, batch_size=config.PHASE1_BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=config.PHASE1_BATCH_SIZE * 2, shuffle=False, num_workers=0, pin_memory=False)

    print(f"Train samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

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

        # Calculate prototypes from ALL training data at the start of epoch for stability
        with torch.no_grad():
            all_train_feat_epoch = []
            all_train_lbl_epoch = []
            # Use a temporary loader to iterate through all training data once
            temp_train_loader = DataLoader(train_dataset, batch_size=config.PHASE1_BATCH_SIZE * 2, shuffle=False, num_workers=0)
            for inputs, labels in temp_train_loader:
                 inputs, labels = inputs.to(config.DEVICE), labels.to(config.DEVICE)
                 features = model(inputs)
                 all_train_feat_epoch.append(features)
                 all_train_lbl_epoch.append(labels)
            all_train_feat_epoch = torch.cat(all_train_feat_epoch)
            all_train_lbl_epoch = torch.cat(all_train_lbl_epoch)
            stable_prototypes_epoch = calculate_prototypes(all_train_feat_epoch, all_train_lbl_epoch, num_domains_phase1)
            del temp_train_loader # Free memory


        for i, (inputs, domain_labels) in enumerate(train_loader):
            inputs, domain_labels = inputs.to(config.DEVICE), domain_labels.to(config.DEVICE)
            optimizer.zero_grad()
            query_features = model(inputs)

            # Use stable prototypes calculated once per epoch
            loss, acc = prototypical_loss(query_features, stable_prototypes_epoch, domain_labels, distance='euclidean', temperature=10.0)

            loss.backward()
            # Gradient clipping can sometimes help stabilize training
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            running_acc += acc.item() # Accuracy of classifying query points to correct domain prototype

            if (i + 1) % max(1, num_batches // 4) == 0:
                 print(f"  Batch {i+1}/{num_batches} | Loss: {loss.item():.4f} | Acc: {acc.item():.4f}")


        epoch_train_loss = running_loss / num_batches
        epoch_train_acc = running_acc / num_batches
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)

        # --- Validation Loop ---
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        all_val_labels_epoch = []
        all_val_preds_epoch = []
        all_val_features_epoch = [] # For t-SNE

        with torch.no_grad():
             # Use the same stable prototypes calculated from training data
            stable_prototypes_val = stable_prototypes_epoch

            for inputs, domain_labels in val_loader:
                inputs, domain_labels = inputs.to(config.DEVICE), domain_labels.to(config.DEVICE)
                query_features = model(inputs)
                loss, acc, preds = prototypical_loss(query_features, stable_prototypes_val, domain_labels, distance='cosine', temperature=10.0, return_preds=True) # Modified loss to return preds
                val_loss += loss.item()
                val_acc += acc.item() # Accuracy based on proto assignment

                all_val_labels_epoch.extend(domain_labels.cpu().numpy())
                all_val_preds_epoch.extend(preds.cpu().numpy())
                all_val_features_epoch.append(query_features.cpu())


        epoch_val_loss = val_loss / len(val_loader)
        epoch_val_acc = val_acc / len(val_loader) # Average accuracy across batches
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)

        # Calculate overall validation accuracy using sklearn
        epoch_val_acc_sk = accuracy_score(all_val_labels_epoch, all_val_preds_epoch)
        print(f"Epoch {epoch+1} Summary: Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.4f} | Val Loss: {epoch_val_loss:.4f} Acc(Proto): {epoch_val_acc:.4f} Acc(SK): {epoch_val_acc_sk:.4f}")


        scheduler.step(epoch_val_loss)
        early_stopping(epoch_val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered for Phase 1.")
            break

    # --- 4. Save Artifacts & Evaluate Final Model ---
    print("\nLoading best Phase 1 model weights...")
    # Load the best model saved by early stopping
    if os.path.exists(config.P1_MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(config.P1_MODEL_SAVE_PATH, map_location=config.DEVICE))
        print(f"Best model loaded from {config.P1_MODEL_SAVE_PATH}")
    else:
        print("Warning: Best model checkpoint not found. Using model from last epoch.")


    print("Calculating and saving final Phase 1 prototypes from training data...")
    model.eval()
    with torch.no_grad():
         # Recalculate features and prototypes using the best model on the full training set
         all_train_feat_final = []
         all_train_lbl_final = []
         temp_train_loader = DataLoader(train_dataset, batch_size=config.PHASE1_BATCH_SIZE * 2, shuffle=False, num_workers=0)
         for inputs, labels in temp_train_loader:
             inputs = inputs.to(config.DEVICE)
             all_train_feat_final.append(model(inputs).cpu()) # Collect features on CPU
             all_train_lbl_final.append(labels.cpu())
         all_train_feat_final = torch.cat(all_train_feat_final).to(config.DEVICE) # Move back to device for proto calc
         all_train_lbl_final = torch.cat(all_train_lbl_final).to(config.DEVICE)
         final_prototypes = calculate_prototypes(all_train_feat_final, all_train_lbl_final, num_domains_phase1)
         del temp_train_loader

    torch.save(final_prototypes, config.P1_PROTOTYPES_SAVE_PATH)
    print(f"Final prototypes saved to {config.P1_PROTOTYPES_SAVE_PATH}")

    # Final Validation Evaluation
    print("\n--- Final Phase 1 Validation Results (Best Model) ---")
    model.eval()
    final_val_loss = 0.0
    final_val_labels = []
    final_val_preds = []
    final_val_features = []
    with torch.no_grad():
        for inputs, domain_labels in val_loader:
            inputs, domain_labels = inputs.to(config.DEVICE), domain_labels.to(config.DEVICE)
            query_features = model(inputs)
            # Use final calculated prototypes
            loss, _, preds = prototypical_loss(query_features, final_prototypes, domain_labels, distance='cosine', temperature=10.0, return_preds=True)
            final_val_loss += loss.item()
            final_val_labels.extend(domain_labels.cpu().numpy())
            final_val_preds.extend(preds.cpu().numpy())
            final_val_features.append(query_features.cpu()) # Collect features on CPU

    final_val_loss /= len(val_loader)
    final_val_features = torch.cat(final_val_features).numpy()

    print(f"Final Validation Loss: {final_val_loss:.4f}")
    report = classification_report(final_val_labels, final_val_preds,
                                   labels=list(domain_reverse_mapping_p1.keys()), # Use actual labels present
                                   target_names=[domain_reverse_mapping_p1.get(i, f"Unknown_{i}") for i in sorted(domain_reverse_mapping_p1.keys())],
                                   zero_division=0)
    print("Final Validation Classification Report (Domains):")
    print(report)
    final_accuracy = accuracy_score(final_val_labels, final_val_preds)
    print(f"Final Validation Accuracy: {final_accuracy:.4f}")


    # Plotting
    plot_training_history(history, title='Phase 1 Base Model Training', save_path=config.P1_HISTORY_PLOT_PATH)

    plot_confusion_matrix_heatmap(final_val_labels, final_val_preds,
                                  class_names=[domain_reverse_mapping_p1.get(i, f"Unknown_{i}") for i in sorted(domain_reverse_mapping_p1.keys())],
                                  title='Phase 1 Validation Confusion Matrix (Domains)',
                                  save_path=config.P1_CONFMAT_PLOT_PATH)

    plot_tsne(final_val_features, final_val_labels, domain_reverse_mapping_p1,
              title='Phase 1 Validation Feature t-SNE (Best Model)',
              save_path=config.P1_TSNE_PLOT_PATH)


    phase1_time = time.time() - start_time_p1
    print(f"--- Training Phase 1 Completed in {phase1_time:.2f} seconds ---")

if __name__ == "__main__":
    # Need to modify prototypical_loss in utils.py to optionally return predictions
    # Add return_preds=False parameter
    # Inside prototypical_loss, before returning loss, acc:
    #   preds = torch.argmax(logits, dim=1)
    #   if return_preds:
    #       return loss, acc, preds
    #   else:
    #       return loss, acc

    # Make sure the modified utils.py is used.
    train_phase1()