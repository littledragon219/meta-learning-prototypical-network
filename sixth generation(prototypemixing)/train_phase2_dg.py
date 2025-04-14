# train_phase2_dg.py
# ... (其他部分保持不变) ...
import os
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset, Subset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import time
import random
import torch.nn as nn

# Import local modules
import config
from datapreprocessing import collect_data_from_folders, calculate_expected_length
from model import FeatureExtractor1DCNN, HandcraftedFeatureExtractor, MaturityClassifier
from utils import (EarlyStopping, calculate_prototypes, prototypical_loss,
                   mix_features, mixed_feature_loss, plot_training_history,
                   plot_confusion_matrix_heatmap, plot_tsne)
try:
    from train_phase2_dg import get_combined_label, get_combined_class_names # Keep this structure
except ImportError: # Fallback definition if run standalone or import fails
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
# --- (create_domain_datasets_p2 and create_episode functions remain the same) ---
def create_domain_datasets_p2(data_paths, domain_mapping, scaler, feature_type, cfg):
    """Loads data for Phase 2 domains and creates separate datasets."""
    domain_datasets = {}
    expected_len = None
    if feature_type == 'cnn':
        expected_len = calculate_expected_length(
            cfg.SIGNAL_START_TIME, cfg.SIGNAL_END_TIME, cfg.SAMPLING_FREQUENCY)
    print("Loading data for Phase 2 domains...")
    for fruit_name, paths in data_paths.items():
        if fruit_name not in domain_mapping: continue
        domain_idx = domain_mapping[fruit_name]
        h_feat, signals, mat_labels, dom_labels = collect_data_from_folders(
            fruit_name=fruit_name, ripe_folder=paths["ripe"], rotten_folder=paths["rotten"],
            domain_label=domain_idx, feature_type=feature_type, expected_length=expected_len,
            cfg=cfg, n_augments=cfg.AUGMENTATIONS_PER_IMAGE )
        if h_feat.size == 0: continue
        if feature_type == 'handcrafted':
            if scaler is None: raise ValueError("Scaler needed for handcrafted features.")
            h_feat_scaled = scaler.transform(h_feat)
        else: h_feat_scaled = h_feat
        if feature_type == 'cnn':
            if signals.size == 0: continue
            X_data = torch.tensor(signals, dtype=torch.float32)
        else: X_data = torch.tensor(h_feat_scaled, dtype=torch.float32)
        y_mat = torch.tensor(mat_labels, dtype=torch.long)
        y_dom = torch.tensor(dom_labels, dtype=torch.long)
        current_dataset = TensorDataset(X_data, y_mat, y_dom)
        domain_datasets[domain_idx] = current_dataset
        print(f"  Domain {fruit_name} ({domain_idx}): {len(current_dataset)} samples")
    if not domain_datasets: raise ValueError("No domain data loaded successfully for Phase 2.")
    return domain_datasets

def create_episode(domain_datasets, domains_in_episode, samples_per_domain):
    """Creates a batch (episode) by sampling from specified domains."""
    episode_data = []
    episode_dom_labels = []
    for domain_idx in domains_in_episode:
        if domain_idx not in domain_datasets: continue
        domain_set = domain_datasets[domain_idx]
        num_domain_samples = len(domain_set)
        if num_domain_samples == 0: continue
        num_to_sample = min(samples_per_domain, num_domain_samples)
        replace = num_to_sample > num_domain_samples
        sampled_indices = np.random.choice(num_domain_samples, num_to_sample, replace=replace)
        for idx in sampled_indices:
            data, _, dom_label = domain_set[idx]
            episode_data.append(data)
            episode_dom_labels.append(dom_label)
    if not episode_data: return None, None
    try:
        episode_data = torch.stack(episode_data)
        episode_dom_labels = torch.stack(episode_dom_labels)
    except Exception as e:
         print(f"Error stacking episode data: {e}")
         return None, None
    return episode_data, episode_dom_labels

def train_phase2_dg():
    # --- (Phase 1 Loading, Data Prep, P2 Setup - remain the same) ---
    print("--- Starting Training Phase 2: DG & S2 Training (Revised Eval) ---")
    start_time_p2 = time.time()
    os.makedirs(config.BASE_SAVE_DIR, exist_ok=True)
    np.random.seed(config.SEED); torch.manual_seed(config.SEED)
    print("Loading Phase 1 artifacts...")
    scaler = None
    if config.FEATURE_TYPE == 'handcrafted':
        try: scaler = joblib.load(config.P1_SCALER_PATH); print(f"Loaded Phase 1 scaler.")
        except FileNotFoundError: print(f"Error: P1 scaler not found."); return
    print("Loading Phase 1 feature extractor model...")
    expected_len_cnn = None
    if config.FEATURE_TYPE == 'cnn': expected_len_cnn = calculate_expected_length(config.SIGNAL_START_TIME, config.SIGNAL_END_TIME, config.SAMPLING_FREQUENCY)
    if config.FEATURE_TYPE == 'cnn': model = FeatureExtractor1DCNN(sequence_length=expected_len_cnn, embedding_dim=config.EMBEDDING_DIM).to(config.DEVICE)
    else: model = HandcraftedFeatureExtractor(input_dim=config.HANDCRAFTED_DIM, embedding_dim=config.EMBEDDING_DIM).to(config.DEVICE)
    try: model.load_state_dict(torch.load(config.P1_MODEL_SAVE_PATH, map_location=config.DEVICE)); print(f"Loaded Phase 1 model weights.")
    except Exception as e: print(f"Error loading Phase 1 model: {e}."); return
    domain_mapping_p2 = {name: i for i, name in enumerate(config.PHASE2_DATA_PATHS.keys())}
    num_domains_p2 = len(domain_mapping_p2)
    domain_reverse_mapping_p2 = {v: k for k, v in domain_mapping_p2.items()}
    domain_datasets = create_domain_datasets_p2(config.PHASE2_DATA_PATHS, domain_mapping_p2, scaler, config.FEATURE_TYPE, config)
    all_datasets = list(domain_datasets.values())
    if not all_datasets: raise ValueError("No datasets created for Phase 2.")
    combined_dataset = ConcatDataset(all_datasets)
    combined_indices = np.arange(len(combined_dataset))
    combined_dom_labels = []
    for ds in all_datasets:
        if isinstance(ds, TensorDataset) and len(ds.tensors) > 2: combined_dom_labels.append(ds.tensors[2].numpy())
        else: print(f"Warning: Problem accessing domain labels from a dataset in Phase 2.")
    if not combined_dom_labels: raise ValueError("Could not extract domain labels for validation split.")
    combined_dom_labels = np.concatenate(combined_dom_labels)
    train_indices, val_indices = train_test_split(combined_indices, test_size=config.PHASE2_VALIDATION_SPLIT, stratify=combined_dom_labels, random_state=config.SEED)
    train_subset = Subset(combined_dataset, train_indices)
    val_subset = Subset(combined_dataset, val_indices)
    val_loader = DataLoader(val_subset, batch_size=config.PHASE2_BATCH_SIZE_PER_DOMAIN * num_domains_p2, shuffle=False, num_workers=0, pin_memory=False)
    print(f"\nPhase 2 Data Split ({num_domains_p2} domains: {list(domain_mapping_p2.keys())}):")
    print(f"  Total Samples: {len(combined_dataset)}, Train: {len(train_subset)}, Val: {len(val_subset)}")
    optimizer = optim.AdamW(model.parameters(), lr=config.PHASE2_LR, weight_decay=config.PHASE2_WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=config.PHASE2_PATIENCE // 2, factor=0.2, verbose=True)
    early_stopping = EarlyStopping(patience=config.PHASE2_PATIENCE, verbose=True, path=config.P2_MODEL_SAVE_PATH)
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc_domain': [], 'val_acc_combined': []}
    num_training_samples = len(train_subset)
    effective_batch_size = config.PHASE2_DOMAINS_PER_EPISODE * config.PHASE2_BATCH_SIZE_PER_DOMAIN
    episodes_per_epoch = max(20, num_training_samples // effective_batch_size)
    print(f"\nStarting Phase 2 training with {episodes_per_epoch} episodes per epoch.")
    available_domain_indices = list(domain_datasets.keys())
    effective_domains_per_episode = min(len(available_domain_indices), config.PHASE2_DOMAINS_PER_EPISODE)

    # --- 3. Episodic Training Loop (Phase 2 DG) ---
    # (No changes needed in the training loop itself)
    for epoch in range(config.PHASE2_EPOCHS):
        print(f"\nPhase 2 Epoch {epoch+1}/{config.PHASE2_EPOCHS}")
        model.train()
        running_loss, running_acc, running_mix_loss = 0.0, 0.0, 0.0
        # Calculate stable prototypes from training subset
        with torch.no_grad():
            train_features_epoch, train_labels_epoch = [], []
            temp_train_loader = DataLoader(train_subset, batch_size=effective_batch_size*2, shuffle=False, num_workers=0)
            for inputs, _, domain_labels in temp_train_loader:
                inputs = inputs.to(config.DEVICE)
                train_features_epoch.append(model(inputs).cpu())
                train_labels_epoch.append(domain_labels.cpu())
            train_features_epoch = torch.cat(train_features_epoch).to(config.DEVICE)
            train_labels_epoch = torch.cat(train_labels_epoch).to(config.DEVICE)
            stable_prototypes_epoch = calculate_prototypes(train_features_epoch, train_labels_epoch, num_domains_p2)
            del temp_train_loader

        # Episodic training steps
        for i in range(episodes_per_epoch):
            if len(available_domain_indices) < effective_domains_per_episode: domains_in_episode = available_domain_indices
            else: domains_in_episode = random.sample(available_domain_indices, effective_domains_per_episode)
            episode_inputs, episode_labels = create_episode(domain_datasets, domains_in_episode, config.PHASE2_BATCH_SIZE_PER_DOMAIN)
            if episode_inputs is None: continue
            episode_inputs, episode_labels = episode_inputs.to(config.DEVICE), episode_labels.to(config.DEVICE)
            optimizer.zero_grad()
            query_features = model(episode_inputs)
            proto_loss, proto_acc = prototypical_loss(query_features, stable_prototypes_epoch, episode_labels, distance='cosine', temperature=10.0)
            mix_loss = torch.tensor(0.0, device=config.DEVICE)
            if config.PHASE2_USE_FEATURE_MIXING and len(episode_inputs) > 1 and len(torch.unique(episode_labels)) > 1:
                mixed_features, mixed_soft_labels = mix_features(query_features, episode_labels, num_domains_p2, num_mixes_per_class=1, alpha=config.PHASE2_MIXING_ALPHA)
                if mixed_features.numel() > 0:
                    mix_loss = mixed_feature_loss(mixed_features.to(config.DEVICE), mixed_soft_labels.to(config.DEVICE), stable_prototypes_epoch, distance='cosine', temperature=10.0)
            total_loss = proto_loss + config.PHASE2_MIXING_LOSS_WEIGHT * mix_loss
            total_loss.backward()
            optimizer.step()
            running_loss += proto_loss.item()
            running_acc += proto_acc.item()
            running_mix_loss += mix_loss.item()

        epoch_train_loss = running_loss / episodes_per_epoch if episodes_per_epoch > 0 else 0
        epoch_train_acc = running_acc / episodes_per_epoch if episodes_per_epoch > 0 else 0
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        print(f"Epoch {epoch+1} Summary: Train Loss: {epoch_train_loss:.4f} | Train Domain Acc: {epoch_train_acc:.4f}")

        # --- Phase 2 Validation Loop (Domain Only Eval) ---
        model.eval()
        val_loss_dom = 0.0
        all_val_labels_dom = []
        all_val_preds_dom = []
        with torch.no_grad():
            stable_prototypes_val = stable_prototypes_epoch
            for inputs, _, domain_labels in val_loader:
                inputs, domain_labels = inputs.to(config.DEVICE), domain_labels.to(config.DEVICE)
                query_features = model(inputs)
                loss, _, preds = prototypical_loss(query_features, stable_prototypes_val, domain_labels, distance='cosine', temperature=10.0, return_preds=True)
                val_loss_dom += loss.item()
                all_val_labels_dom.extend(domain_labels.cpu().numpy())
                all_val_preds_dom.extend(preds.cpu().numpy())
        epoch_val_loss = val_loss_dom / len(val_loader) if len(val_loader) > 0 else 0
        epoch_val_acc_domain = accuracy_score(all_val_labels_dom, all_val_preds_dom) if all_val_labels_dom else 0
        history['val_loss'].append(epoch_val_loss)
        history['val_acc_domain'].append(epoch_val_acc_domain)
        history['val_acc_combined'].append(0.0)
        print(f"Epoch {epoch+1} Validation: Val Loss: {epoch_val_loss:.4f} | Val Domain Acc: {epoch_val_acc_domain:.4f}")

        scheduler.step(epoch_val_loss)
        early_stopping(epoch_val_loss, model)
        if early_stopping.early_stop: print("Early stopping triggered for Phase 2."); break

    # --- 4. Save Final Phase 2 Artifacts ---
    # (Save model and prototypes - remain the same)
    print("\nLoading best Phase 2 model weights...")
    if os.path.exists(config.P2_MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(config.P2_MODEL_SAVE_PATH, map_location=config.DEVICE))
        print(f"Best P2 model loaded from {config.P2_MODEL_SAVE_PATH}")
    else: print("Warning: Best Phase 2 model checkpoint not found.")
    print("Calculating and saving final Phase 2 prototypes from training subset...")
    model.eval()
    with torch.no_grad(): # Recalculate prototypes using the best model
        final_train_features, final_train_labels = [], []
        temp_train_loader = DataLoader(train_subset, batch_size=effective_batch_size*2, shuffle=False, num_workers=0)
        for inputs, _, domain_labels in temp_train_loader:
            inputs = inputs.to(config.DEVICE)
            final_train_features.append(model(inputs).cpu())
            final_train_labels.append(domain_labels.cpu())
        final_train_features = torch.cat(final_train_features).to(config.DEVICE)
        final_train_labels = torch.cat(final_train_labels).to(config.DEVICE)
        final_prototypes_p2 = calculate_prototypes(final_train_features, final_train_labels, num_domains_p2)
        del temp_train_loader
    torch.save(final_prototypes_p2, config.P2_PROTOTYPES_SAVE_PATH)
    print(f"Final Phase 2 prototypes saved to {config.P2_PROTOTYPES_SAVE_PATH}")
    history['val_acc'] = history['val_acc_domain']

    plot_training_history(history, title='Phase 2 DG Training History', save_path=config.P2_HISTORY_PLOT_PATH)
    phase2_feat_time = time.time() - start_time_p2
    print(f"--- Training Phase 2 (Feature Extractor) Completed in {phase2_feat_time:.2f} seconds ---")


    # --- 5. Train Stage 2 Maturity Classifiers ---
    # (This section remains the same - it trains S2 heads based on TRUE domain labels)
    print("\n--- Starting Stage 2 Maturity Classifier Training ---")
    start_time_s2 = time.time()
    model.eval() # Freeze the loaded best P2 feature extractor
    for param in model.parameters(): param.requires_grad = False
    print("Phase 2 Feature Extractor frozen.")
    print("Extracting features for Stage 2 training/validation...")
    train_features_s2, train_mat_labels_s2, train_dom_labels_s2 = [], [], []
    val_features_s2, val_mat_labels_s2, val_dom_labels_s2 = [], [], []
    with torch.no_grad():
        temp_train_loader_s2 = DataLoader(train_subset, batch_size=effective_batch_size*2, shuffle=False, num_workers=0)
        temp_val_loader_s2 = DataLoader(val_subset, batch_size=effective_batch_size*2, shuffle=False, num_workers=0) # Use val_subset for validation features
        for inputs, mat_labels, dom_labels in temp_train_loader_s2:
             train_features_s2.append(model(inputs.to(config.DEVICE)).cpu())
             train_mat_labels_s2.append(mat_labels)
             train_dom_labels_s2.append(dom_labels)
        for inputs, mat_labels, dom_labels in temp_val_loader_s2:
             val_features_s2.append(model(inputs.to(config.DEVICE)).cpu())
             val_mat_labels_s2.append(mat_labels)
             val_dom_labels_s2.append(dom_labels)
    train_features_s2 = torch.cat(train_features_s2)
    train_mat_labels_s2 = torch.cat(train_mat_labels_s2)
    train_dom_labels_s2 = torch.cat(train_dom_labels_s2)
    val_features_s2 = torch.cat(val_features_s2)
    val_mat_labels_s2 = torch.cat(val_mat_labels_s2)
    val_dom_labels_s2 = torch.cat(val_dom_labels_s2)
    del temp_train_loader_s2, temp_val_loader_s2

    stage2_maturity_models = {} # Store trained S2 models here
    for domain_idx, domain_name in domain_reverse_mapping_p2.items():
        print(f"\n--- Training Stage 2 Maturity Classifier for: {domain_name} (Domain {domain_idx}) ---")
        train_mask = (train_dom_labels_s2 == domain_idx); val_mask = (val_dom_labels_s2 == domain_idx)
        if not torch.any(train_mask) or not torch.any(val_mask): print(f"Skipping {domain_name}: Not enough samples."); continue
        X_train_dom, y_train_dom = train_features_s2[train_mask], train_mat_labels_s2[train_mask]
        X_val_dom, y_val_dom = val_features_s2[val_mask], val_mat_labels_s2[val_mask]
        if X_train_dom.shape[0] == 0 or X_val_dom.shape[0] == 0: print(f"Skipping {domain_name}: Zero samples after filtering."); continue
        train_dataset_s2_dom = TensorDataset(X_train_dom, y_train_dom); val_dataset_s2_dom = TensorDataset(X_val_dom, y_val_dom)
        train_loader_s2_dom = DataLoader(train_dataset_s2_dom, batch_size=config.STAGE2_BATCH_SIZE, shuffle=True, num_workers=0)
        val_loader_s2_dom = DataLoader(val_dataset_s2_dom, batch_size=config.STAGE2_BATCH_SIZE * 2, shuffle=False, num_workers=0)
        print(f"  Domain {domain_name} - Train: {len(train_dataset_s2_dom)}, Val: {len(val_dataset_s2_dom)}")
        model_s2 = MaturityClassifier(input_dim=config.EMBEDDING_DIM, hidden_dim=32, num_classes=2).to(config.DEVICE)
        optimizer_s2 = optim.AdamW(model_s2.parameters(), lr=config.STAGE2_LR, weight_decay=config.STAGE2_WEIGHT_DECAY)
        scheduler_s2 = optim.lr_scheduler.ReduceLROnPlateau(optimizer_s2, 'min', patience=config.STAGE2_PATIENCE // 2, factor=0.2, verbose=False)
        criterion_s2 = nn.CrossEntropyLoss()
        s2_model_path = config.S2_MODEL_SAVE_TEMPLATE.format(domain_name)
        early_stopping_s2 = EarlyStopping(patience=config.STAGE2_PATIENCE, verbose=False, path=s2_model_path) # Less verbose S2 stopping
        history_s2 = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        # S2 Training Loop... (kept concise for brevity, same as before)
        for epoch in range(config.STAGE2_EPOCHS):
            model_s2.train(); tr_loss, tr_acc = 0.0, 0.0
            for features, labels in train_loader_s2_dom: features, labels = features.to(config.DEVICE), labels.to(config.DEVICE); optimizer_s2.zero_grad(); logits = model_s2(features); loss = criterion_s2(logits, labels); loss.backward(); optimizer_s2.step(); tr_loss += loss.item(); tr_acc += (torch.argmax(logits, dim=1) == labels).float().mean().item()
            epoch_train_loss = tr_loss / len(train_loader_s2_dom) if len(train_loader_s2_dom)>0 else 0; epoch_train_acc = tr_acc / len(train_loader_s2_dom) if len(train_loader_s2_dom)>0 else 0
            history_s2['train_loss'].append(epoch_train_loss); history_s2['train_acc'].append(epoch_train_acc)
            model_s2.eval(); v_loss, v_acc = 0.0, 0.0; all_s2_val_labels, all_s2_val_preds = [], []
            with torch.no_grad():
                for features, labels in val_loader_s2_dom: features, labels = features.to(config.DEVICE), labels.to(config.DEVICE); logits = model_s2(features); loss = criterion_s2(logits, labels); v_loss += loss.item(); preds = torch.argmax(logits, dim=1); v_acc += (preds == labels).float().mean().item(); all_s2_val_labels.extend(labels.cpu().numpy()); all_s2_val_preds.extend(preds.cpu().numpy())
            epoch_val_loss = v_loss / len(val_loader_s2_dom) if len(val_loader_s2_dom)>0 else 0; epoch_val_acc = v_acc / len(val_loader_s2_dom) if len(val_loader_s2_dom)>0 else 0
            history_s2['val_loss'].append(epoch_val_loss); history_s2['val_acc'].append(epoch_val_acc)
            if (epoch + 1) % 15 == 0: print(f"    S2 Epoch {epoch+1} Val Loss: {epoch_val_loss:.4f}") # Less frequent printing
            scheduler_s2.step(epoch_val_loss); early_stopping_s2(epoch_val_loss, model_s2)
            if early_stopping_s2.early_stop: print(f"    Early stopping S2 {domain_name}."); break
        # Load best S2 model and store it
        if os.path.exists(s2_model_path):
             model_s2.load_state_dict(torch.load(s2_model_path, map_location=config.DEVICE))
             stage2_maturity_models[domain_idx] = model_s2 # Store the best trained S2 model
             print(f"  Stored best S2 model for {domain_name}")
        else: print(f"  Warning: Best S2 model for {domain_name} not saved/found.")
        s2_hist_path = config.S2_HISTORY_PLOT_TEMPLATE.format(domain_name)
        plot_training_history(history_s2, f'S2 Maturity Training ({domain_name})', save_path=s2_hist_path)
        s2_cm_path = config.S2_CONFMAT_PLOT_TEMPLATE.format(domain_name)
        if all_s2_val_labels: plot_confusion_matrix_heatmap(all_s2_val_labels, all_s2_val_preds, ["Ripe", "Rotten"], f'S2 Val CM ({domain_name})', s2_cm_path)

    stage2_time = time.time() - start_time_s2
    print(f"\n--- Stage 2 Maturity Training Completed in {stage2_time:.2f} seconds ---")

    # --- 6. Final Phase 2 Validation (Combined Domain + Maturity Prediction) ---
    print("\n--- Final Phase 2 Validation Results (Combined Prediction) ---")

    # <<< START: Added check for S2 models before final validation >>>
    if not stage2_maturity_models:
        print("ERROR: No Stage 2 maturity models were loaded successfully. Cannot perform combined validation.")
        return # Exit if no S2 models are available
    print(f"Loaded {len(stage2_maturity_models)} Stage 2 maturity models for final validation.")
    # <<< END: Added check >>>

    model.eval() # Best P2 feature extractor
    all_final_val_true_dom = []
    all_final_val_true_mat = []
    all_final_val_pred_dom = []
    all_final_val_pred_mat = [] # THIS IS THE LIST TO CHECK
    all_final_val_features = []

    missing_s2_warnings_final = set()
    with torch.no_grad():
        for batch_idx, (inputs, true_mat, true_dom) in enumerate(val_loader):
            inputs, true_mat, true_dom = inputs.to(config.DEVICE), true_mat.to(config.DEVICE), true_dom.to(config.DEVICE)

            # Step 1: Get P2 features and predict domain
            features = model(inputs)
            all_final_val_features.append(features.cpu())

            # Ensure prototypes are valid before prediction
            if final_prototypes_p2 is None or final_prototypes_p2.shape[0] != num_domains_p2:
                 print(f"Error: Invalid P2 prototypes for final validation (Shape: {final_prototypes_p2.shape if final_prototypes_p2 is not None else 'None'}). Skipping batch {batch_idx}.")
                 # Append dummy values or skip batch? Skipping is safer.
                 continue # Skip this batch

            _, _, predicted_dom = prototypical_loss(features, final_prototypes_p2, true_dom, distance='cosine', temperature=10.0, return_preds=True)

            all_final_val_true_dom.extend(true_dom.cpu().numpy())
            all_final_val_pred_dom.extend(predicted_dom.cpu().numpy())
            all_final_val_true_mat.extend(true_mat.cpu().numpy())

            # Step 2: Predict maturity using predicted domain's S2 head
            batch_preds_maturity = []
            for i in range(features.shape[0]):
                sample_feature = features[i:i+1]
                pred_dom_idx = predicted_dom[i].item()

                if pred_dom_idx in stage2_maturity_models:
                    s2_model = stage2_maturity_models[pred_dom_idx]
                    s2_model.eval()
                    try:
                        maturity_logits = s2_model(sample_feature)
                        pred_mat = torch.argmax(maturity_logits, dim=1).item()
                        batch_preds_maturity.append(pred_mat)
                    except Exception as e_s2:
                         print(f"Error during S2 prediction for predicted domain {pred_dom_idx}, sample {i} in batch {batch_idx}: {e_s2}")
                         batch_preds_maturity.append(0) # Fallback on error
                else:
                    if pred_dom_idx not in missing_s2_warnings_final:
                         print(f"Warning: S2 model missing for predicted domain {pred_dom_idx} ({domain_reverse_mapping_p2.get(pred_dom_idx,'?')}) during final validation. Predicting 0.")
                         missing_s2_warnings_final.add(pred_dom_idx)
                    batch_preds_maturity.append(0)
            # <<< START: Debug print for batch_preds_maturity length >>>
            # print(f"Debug Batch {batch_idx}: true_mat len={len(true_mat)}, batch_preds_maturity len={len(batch_preds_maturity)}")
            # <<< END: Debug print >>>
            all_final_val_pred_mat.extend(batch_preds_maturity) # Extend the list for predicted maturity

    # <<< START: Length check before combined evaluation >>>
    print("\n--- Length Check Before Combined Evaluation ---")
    len_true_dom = len(all_final_val_true_dom)
    len_pred_dom = len(all_final_val_pred_dom)
    len_true_mat = len(all_final_val_true_mat)
    len_pred_mat = len(all_final_val_pred_mat) # Check this length
    print(f"Length of true domain labels: {len_true_dom}")
    print(f"Length of predicted domain labels: {len_pred_dom}")
    print(f"Length of true maturity labels: {len_true_mat}")
    print(f"Length of predicted maturity labels: {len_pred_mat}")

    if len_true_dom == 0 or len_true_mat == 0:
        print("Validation data seems empty. Skipping final evaluation.")
        return # Cant evaluate if no true labels

    # Check if prediction list lengths match true list lengths
    proceed_with_combined_eval = True
    if len_pred_dom != len_true_dom:
        print(f"ERROR: Length mismatch for domain labels! True={len_true_dom}, Pred={len_pred_dom}")
        proceed_with_combined_eval = False # Cannot proceed reliably
    if len_pred_mat != len_true_mat:
        print(f"ERROR: Length mismatch for maturity labels! True={len_true_mat}, Pred={len_pred_mat}")
        # This was the likely cause of the original error if len_pred_mat was 0
        proceed_with_combined_eval = False # Cannot proceed reliably

    # <<< END: Length check >>>

    # --- Evaluate Domain Prediction Accuracy ---
    if len_true_dom > 0 and len_pred_dom == len_true_dom:
        print("\n--- Domain Classification Report (Validation - P2 Model) ---")
        domain_labels_list = sorted(domain_reverse_mapping_p2.keys())
        domain_target_names=[domain_reverse_mapping_p2.get(i, f"Unknown_{i}") for i in domain_labels_list]
        try:
            report_dom = classification_report(all_final_val_true_dom, all_final_val_pred_dom, labels=domain_labels_list, target_names=domain_target_names, zero_division=0)
            print(report_dom)
            accuracy_dom = accuracy_score(all_final_val_true_dom, all_final_val_pred_dom)
            print(f"Final Validation Domain Accuracy: {accuracy_dom:.4f}")
            plot_confusion_matrix_heatmap(all_final_val_true_dom, all_final_val_pred_dom, domain_target_names, 'P2 Validation CM (Domains)', config.P2_VAL_DOMAIN_CONFMAT_PLOT_PATH)
        except ValueError as e_report:
             print(f"Error generating domain classification report/plot: {e_report}")
             print(f"Unique true domain labels: {np.unique(all_final_val_true_dom)}")
             print(f"Unique predicted domain labels: {np.unique(all_final_val_pred_dom)}")
    else:
        print("Skipping Domain Classification Report due to length mismatch or zero length.")


    # --- Evaluate Combined Prediction Accuracy ---
    if proceed_with_combined_eval and len_true_mat > 0: # Check flag and ensure true mat list has data
        print("\n--- Combined Prediction Report (Validation - P2 Model + S2 Heads) ---")
        num_mat_classes = 2
        # <<< START: Wrap list comprehensions in try-except or add length checks >>>
        try:
            true_combined_labels = [get_combined_label(d, m, num_mat_classes) for d, m in zip(all_final_val_true_dom, all_final_val_true_mat)]
            # Check if the input lists for pred_combined_labels have matching, non-zero lengths
            if len(all_final_val_pred_dom) == len(all_final_val_pred_mat) and len(all_final_val_pred_mat) > 0:
                 pred_combined_labels = [get_combined_label(d, m, num_mat_classes) for d, m in zip(all_final_val_pred_dom, all_final_val_pred_mat)]
            else:
                 print("ERROR: Cannot create combined predicted labels due to input list length mismatch or empty predicted maturity list.")
                 pred_combined_labels = [] # Ensure it's empty if inputs are bad

            combined_class_names = get_combined_class_names(domain_reverse_mapping_p2)
            combined_labels_list = list(range(len(combined_class_names)))

            # Final check before metrics calculation
            if len(true_combined_labels) == len(pred_combined_labels) and len(true_combined_labels) > 0:
                 report_comb = classification_report(true_combined_labels, pred_combined_labels, labels=combined_labels_list, target_names=combined_class_names, zero_division=0)
                 print(report_comb)
                 accuracy_comb = accuracy_score(true_combined_labels, pred_combined_labels)
                 print(f"Final Validation Combined Accuracy: {accuracy_comb:.4f}")
                 plot_confusion_matrix_heatmap(true_combined_labels, pred_combined_labels, combined_class_names, 'P2 Validation CM (Combined Domain-Maturity)', config.P2_VAL_COMBINED_CONFMAT_PLOT_PATH)
            else:
                 print("Skipping combined metrics calculation due to inconsistent label list lengths.")

        except Exception as e_comb:
            print(f"An error occurred during combined label creation or evaluation: {e_comb}")
        # <<< END: Wrap list comprehensions >>>
    else:
        print("Skipping Combined Prediction Report due to earlier errors or length mismatches.")

    # --- Plot t-SNE ---
    if all_final_val_features:
        try:
            all_final_val_features = torch.cat(all_final_val_features).numpy()
            if len(all_final_val_features) == len(all_final_val_true_dom): # Ensure features match labels
                plot_tsne(all_final_val_features, all_final_val_true_dom, domain_reverse_mapping_p2, # Color by TRUE domain
                          title='Phase 2 Validation Feature t-SNE (Best P2 Model)',
                          save_path=config.P2_VAL_TSNE_PLOT_PATH)
            else:
                 print(f"Skipping t-SNE plot: Feature length ({len(all_final_val_features)}) doesn't match label length ({len(all_final_val_true_dom)}).")
        except Exception as e_tsne:
             print(f"Error during t-SNE plotting: {e_tsne}")


    print(f"\n--- Phase 2 (DG + S2 Training + Combined Eval) Completed ---")


if __name__ == "__main__":
    train_phase2_dg()