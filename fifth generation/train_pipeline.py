# train_pipeline.py
import os
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import time
import torch.nn as nn


# Import from local modules
from datapreprocessing import (
    collect_data_from_folders, calculate_expected_length,
    DEFAULT_SIGNAL_START_TIME, DEFAULT_SIGNAL_END_TIME, DEFAULT_SAMPLING_FREQUENCY,
    DEFAULT_AUGMENTATIONS
)
from model import (
    FeatureExtractor1DCNN, HandcraftedFeatureExtractor,
    ProtoNetDomainClassifier, MaturityClassifier
)
from utils import (
    EarlyStopping, calculate_prototypes, prototypical_loss,
    plot_training_history, plot_confusion_matrix_heatmap, plot_tsne
)

# --- Configuration ---
SEED = 42
BASE_SAVE_DIR = "pipeline_models_v2"
os.makedirs(BASE_SAVE_DIR, exist_ok=True)

# Data Paths (Update with your actual training data paths)
DATA_PATHS = {
    "grape": {"ripe": r"D:/大二下学期/CDIO/grape_ripemetalearning", "rotten": r"D:/大二下学期/CDIO/grape_rottenmetalearning"},
    "strawberry": {"ripe": r"D:/大二下学期/CDIO/strawberry_ripe", "rotten": r"D:/大二下学期/CDIO/strawberry_rotten"},
    "tomato": {"ripe": r"D:/大二下学期/CDIO/ctomato/tomato_ripe", "rotten": r"D:/大二下学期/CDIO/ctomato/tomato_rotten"},
}
DOMAIN_MAPPING = {name: i for i, name in enumerate(DATA_PATHS.keys())}
DOMAIN_REVERSE_MAPPING = {i: name for name, i in DOMAIN_MAPPING.items()}
NUM_DOMAINS = len(DATA_PATHS)

# Model & Feature Choice
FEATURE_TYPE = 'handcrafted' # Choose 'cnn', 'handcrafted'
if FEATURE_TYPE == 'cnn':
    EMBEDDING_DIM = 128 # Output dimension of the feature extractor
elif FEATURE_TYPE == 'handcrafted':
    EMBEDDING_DIM = 64
else:
    raise ValueError("Invalid FEATURE_TYPE. Choose 'cnn' or 'handcrafted'.")

# Training Hyperparameters
EPOCHS_S1 = 200
EPOCHS_S2 = 150
BATCH_SIZE = 16
LEARNING_RATE_S1 = 1e-4
LEARNING_RATE_S2 = 5e-4
WEIGHT_DECAY = 1e-5
EARLY_STOPPING_PATIENCE_S1 = 20
EARLY_STOPPING_PATIENCE_S2 = 15
VALIDATION_SPLIT_SIZE = 0.15 # Proportion of train data used for validation
TEST_SPLIT_SIZE = 0.15 # Proportion of total data used for final testing

# File Paths for Saving
SCALER_SAVE_PATH = os.path.join(BASE_SAVE_DIR, 'scaler.joblib')
TEST_DATA_SAVE_PATH = os.path.join(BASE_SAVE_DIR, 'test_data.npz')
S1_MODEL_SAVE_PATH = os.path.join(BASE_SAVE_DIR, f'stage1_feature_extractor_{FEATURE_TYPE}_best.pth')
S1_PROTOTYPES_SAVE_PATH = os.path.join(BASE_SAVE_DIR, f'stage1_domain_prototypes_{FEATURE_TYPE}.pth')
S2_MODEL_SAVE_TEMPLATE = os.path.join(BASE_SAVE_DIR, f'stage2_maturity_classifier_{{}}_using_{FEATURE_TYPE}_best.pth') # {} for domain name
S1_HISTORY_PLOT_PATH = os.path.join(BASE_SAVE_DIR, f'stage1_training_history_{FEATURE_TYPE}.png')
S2_HISTORY_PLOT_TEMPLATE = os.path.join(BASE_SAVE_DIR, f'stage2_training_history_{{}}_using_{FEATURE_TYPE}.png') # {} for domain name

# Set Seed for Reproducibility
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    # Potentially make CuDNN deterministic (can slow things down)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- 1. Data Loading and Preparation ---
print("--- Starting Data Loading and Preparation ---")
start_time = time.time()

# Calculate expected signal length based on defaults
EXPECTED_SIGNAL_LENGTH = calculate_expected_length(
    DEFAULT_SIGNAL_START_TIME, DEFAULT_SIGNAL_END_TIME, DEFAULT_SAMPLING_FREQUENCY
)
print(f"Expected signal length: {EXPECTED_SIGNAL_LENGTH}")

all_h_features = []
all_signals = []
all_mat_labels = []
all_dom_labels = []

for fruit_name, paths in DATA_PATHS.items():
    domain_idx = DOMAIN_MAPPING[fruit_name]
    h_feat, signals, mat_labels, dom_labels = collect_data_from_folders(
        fruit_name=fruit_name,
        ripe_folder=paths["ripe"],
        rotten_folder=paths["rotten"],
        domain_label=domain_idx,
        expected_length=EXPECTED_SIGNAL_LENGTH,
        n_augments=DEFAULT_AUGMENTATIONS
    )
    if h_feat.size > 0:
        all_h_features.append(h_feat)
        all_signals.append(signals)
        all_mat_labels.append(mat_labels)
        all_dom_labels.append(dom_labels)

if not all_h_features:
    raise ValueError("No data collected. Check DATA_PATHS and data folders.")

# Concatenate data from all fruits
X_handcrafted = np.concatenate(all_h_features, axis=0)
X_signals = np.concatenate(all_signals, axis=0)
y_maturity = np.concatenate(all_mat_labels, axis=0)
y_domain = np.concatenate(all_dom_labels, axis=0)

print(f"\nTotal samples collected (incl. augmentations): {len(y_domain)}")
print(f"Handcrafted features shape: {X_handcrafted.shape}")
print(f"Signals shape: {X_signals.shape}")
print(f"Maturity labels shape: {y_maturity.shape}")
print(f"Domain labels shape: {y_domain.shape}")
print(f"Domain distribution: {np.bincount(y_domain)}")
print(f"Maturity distribution: {np.bincount(y_maturity)}")


# Split into Train+Validation and Test sets (Stratified by Domain)
indices = np.arange(len(y_domain))
X_trainval_h, X_test_h, \
X_trainval_s, X_test_s, \
y_trainval_m, y_test_m, \
y_trainval_d, y_test_d, \
indices_trainval, indices_test = train_test_split(
    X_handcrafted, X_signals, y_maturity, y_domain, indices,
    test_size=TEST_SPLIT_SIZE, stratify=y_domain, random_state=SEED
)

# Split Train+Validation into Train and Validation sets (Stratified by Domain)
X_train_h, X_val_h, \
X_train_s, X_val_s, \
y_train_m, y_val_m, \
y_train_d, y_val_d = train_test_split(
    X_trainval_h, X_trainval_s, y_trainval_m, y_trainval_d,
    test_size=VALIDATION_SPLIT_SIZE / (1 - TEST_SPLIT_SIZE), # Adjust proportion
    stratify=y_trainval_d, random_state=SEED
)

print("\nDataset Split Sizes:")
print(f"  Train:      {len(y_train_d)}")
print(f"  Validation: {len(y_val_d)}")
print(f"  Test:       {len(y_test_d)}")

# Scale Handcrafted Features (Fit ONLY on Training data)
scaler = StandardScaler()
X_train_h_scaled = scaler.fit_transform(X_train_h)
X_val_h_scaled = scaler.transform(X_val_h)
X_test_h_scaled = scaler.transform(X_test_h)
joblib.dump(scaler, SCALER_SAVE_PATH)
print(f"\nStandardScaler fitted on training data and saved to {SCALER_SAVE_PATH}")

# Save the Test Set for later use by the testing script
np.savez(TEST_DATA_SAVE_PATH,
         X_test_h=X_test_h_scaled, # Save scaled handcrafted features
         X_test_s=X_test_s,       # Save raw signals
         y_test_m=y_test_m,
         y_test_d=y_test_d)
print(f"Test set data saved to {TEST_DATA_SAVE_PATH}")


# Create PyTorch Datasets and DataLoaders
def create_dataset(X_h, X_s, y_m, y_d):
    # Convert to tensors
    tensor_h = torch.tensor(X_h, dtype=torch.float32)
    tensor_s = torch.tensor(X_s, dtype=torch.float32).unsqueeze(1) # Add channel dim for CNN
    tensor_m = torch.tensor(y_m, dtype=torch.long)
    tensor_d = torch.tensor(y_d, dtype=torch.long)
    return TensorDataset(tensor_h, tensor_s, tensor_m, tensor_d)

train_dataset = create_dataset(X_train_h_scaled, X_train_s, y_train_m, y_train_d)
val_dataset = create_dataset(X_val_h_scaled, X_val_s, y_val_m, y_val_d)
# Test dataset is saved, but we can create a loader for potential evaluation here
test_dataset = create_dataset(X_test_h_scaled, X_test_s, y_test_m, y_test_d)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=0, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE * 2, shuffle=False)

data_prep_time = time.time() - start_time
print(f"--- Data Preparation Completed in {data_prep_time:.2f} seconds ---")

# --- 2. Stage 1: Train Prototypical Network for Domain Classification ---
print("\n--- Starting Stage 1: Domain Classification Training ---")
start_time_s1 = time.time()

# Instantiate Feature Extractor based on choice
if FEATURE_TYPE == 'cnn':
    feature_extractor_s1 = FeatureExtractor1DCNN(
        input_channels=1,
        sequence_length=EXPECTED_SIGNAL_LENGTH,
        embedding_dim=EMBEDDING_DIM
    ).to(device)
    input_data_index = 1 # Use signal data (index 1 in dataset tuple)
    print("Using 1D CNN Feature Extractor for Stage 1.")
elif FEATURE_TYPE == 'handcrafted':
     num_handcrafted_features = X_train_h_scaled.shape[1]
     feature_extractor_s1 = HandcraftedFeatureExtractor(
        input_dim=num_handcrafted_features,
        hidden_dim=32, # Can be tuned
        embedding_dim=EMBEDDING_DIM
     ).to(device)
     input_data_index = 0 # Use handcrafted features (index 0 in dataset tuple)
     print("Using Handcrafted Feature Extractor (MLP) for Stage 1.")

# The ProtoNet model is just the feature extractor
model_s1 = feature_extractor_s1

optimizer_s1 = optim.AdamW(model_s1.parameters(), lr=LEARNING_RATE_S1, weight_decay=WEIGHT_DECAY)
# Scheduler reduces LR if validation loss plateaus
scheduler_s1 = optim.lr_scheduler.ReduceLROnPlateau(optimizer_s1, 'min', patience=EARLY_STOPPING_PATIENCE_S1 // 2, factor=0.2, verbose=True)
early_stopping_s1 = EarlyStopping(patience=EARLY_STOPPING_PATIENCE_S1, verbose=True, path=S1_MODEL_SAVE_PATH)

history_s1 = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

# --- Pre-calculate prototypes on the full training set (for validation) ---
# This requires one pass through the training data before the first epoch validation
# Or can be updated each epoch
def get_all_train_features(model, loader, data_idx, device):
    model.eval()
    all_features = []
    all_labels = []
    with torch.no_grad():
        for batch in loader:
            inputs = batch[data_idx].to(device)
            labels = batch[3].to(device) # Domain labels are at index 3
            features = model(inputs)
            all_features.append(features.cpu())
            all_labels.append(labels.cpu())
    model.train()
    return torch.cat(all_features), torch.cat(all_labels)

# --- Stage 1 Training Loop ---
for epoch in range(EPOCHS_S1):
    print(f"\nStage 1 Epoch {epoch+1}/{EPOCHS_S1}")
    model_s1.train()
    running_loss = 0.0
    running_acc = 0.0
    num_batches = len(train_loader)

    # Calculate stable prototypes based on current model state over training set
    # This is computationally more expensive per epoch but more stable
    # If too slow, calculate only every few epochs or use batch prototypes for training loss
    with torch.no_grad():
         all_train_feats_epoch, all_train_labels_epoch = get_all_train_features(model_s1, train_loader, input_data_index, device)
         stable_prototypes_epoch = calculate_prototypes(all_train_feats_epoch.to(device), all_train_labels_epoch.to(device), NUM_DOMAINS)

    for i, batch in enumerate(train_loader):
        inputs = batch[input_data_index].to(device)
        domain_labels = batch[3].to(device) # Domain labels index = 3

        optimizer_s1.zero_grad()

        # Get features (query features for this batch)
        query_features = model_s1(inputs)

        # Calculate loss using stable prototypes calculated for the epoch
        loss, acc = prototypical_loss(query_features, stable_prototypes_epoch, domain_labels, distance='cosine') # Or 'cosine'

        loss.backward()
        optimizer_s1.step()

        running_loss += loss.item()
        running_acc += acc.item()

        if (i + 1) % max(1, num_batches // 4) == 0: # Print progress 4 times per epoch
             print(f"  Batch {i+1}/{num_batches} | Train Loss: {loss.item():.4f} | Train Acc: {acc.item():.4f}")

    epoch_train_loss = running_loss / num_batches
    epoch_train_acc = running_acc / num_batches
    history_s1['train_loss'].append(epoch_train_loss)
    history_s1['train_acc'].append(epoch_train_acc)
    print(f"Epoch {epoch+1} Summary: Avg Train Loss: {epoch_train_loss:.4f} | Avg Train Acc: {epoch_train_acc:.4f}")


    # --- Stage 1 Validation Loop ---
    model_s1.eval()
    val_loss = 0.0
    val_acc = 0.0
    num_val_batches = len(val_loader)

    # Use stable prototypes from the *training* set for validation classification
    with torch.no_grad():
        # If prototypes weren't calculated above, calculate them here using train data
        # all_train_feats_val, all_train_labels_val = get_all_train_features(model_s1, train_loader, input_data_index, device)
        # stable_prototypes_val = calculate_prototypes(all_train_feats_val.to(device), all_train_labels_val.to(device), NUM_DOMAINS)
        stable_prototypes_val = stable_prototypes_epoch # Reuse if calculated above

        for batch in val_loader:
            inputs = batch[input_data_index].to(device)
            domain_labels = batch[3].to(device)
            query_features = model_s1(inputs)

            # Calculate validation loss and accuracy against stable prototypes
            loss, acc = prototypical_loss(query_features, stable_prototypes_val, domain_labels, distance='euclidean')
            val_loss += loss.item()
            val_acc += acc.item()

    epoch_val_loss = val_loss / num_val_batches
    epoch_val_acc = val_acc / num_val_batches
    history_s1['val_loss'].append(epoch_val_loss)
    history_s1['val_acc'].append(epoch_val_acc)

    print(f"Epoch {epoch+1} Validation: Avg Val Loss: {epoch_val_loss:.4f} | Avg Val Acc: {epoch_val_acc:.4f}")

    # Scheduler Step
    scheduler_s1.step(epoch_val_loss)

    # Early Stopping Check
    early_stopping_s1(epoch_val_loss, model_s1)
    if early_stopping_s1.early_stop:
        print("Early stopping triggered for Stage 1.")
        break

# Load the best model state found during early stopping
print(f"Loading best Stage 1 model weights from {S1_MODEL_SAVE_PATH}")
model_s1.load_state_dict(torch.load(S1_MODEL_SAVE_PATH))

# Calculate and save final stable prototypes using the best model
print("Calculating and saving final stable prototypes...")
final_train_features, final_train_labels = get_all_train_features(model_s1, train_loader, input_data_index, device)
final_prototypes = calculate_prototypes(final_train_features.to(device), final_train_labels.to(device), NUM_DOMAINS)
torch.save(final_prototypes, S1_PROTOTYPES_SAVE_PATH)
print(f"Final prototypes saved to {S1_PROTOTYPES_SAVE_PATH}")


# Plot Stage 1 Training History
plot_training_history(history_s1, title='Stage 1 Domain Classifier Training', save_path=S1_HISTORY_PLOT_PATH)

s1_train_time = time.time() - start_time_s1
print(f"--- Stage 1 Training Completed in {s1_train_time:.2f} seconds ---")


# --- 3. Stage 2: Train Maturity Classifiers per Domain ---
print("\n--- Starting Stage 2: Maturity Classification Training ---")
start_time_s2 = time.time()

# Freeze the Stage 1 feature extractor
model_s1.eval()
for param in model_s1.parameters():
    param.requires_grad = False
print("Stage 1 Feature Extractor frozen.")

stage2_histories = {}

for domain_idx, domain_name in DOMAIN_REVERSE_MAPPING.items():
    print(f"\n--- Training Stage 2 Classifier for Domain: {domain_name} (ID: {domain_idx}) ---")

    # Create domain-specific datasets and loaders
    train_mask = (y_train_d == domain_idx)
    val_mask = (y_val_d == domain_idx)

    if not np.any(train_mask) or not np.any(val_mask):
        print(f"Skipping domain {domain_name}: Not enough train or validation samples.")
        continue

    # Select appropriate features based on FEATURE_TYPE
    if FEATURE_TYPE == 'cnn':
        X_train_dom_input = X_train_s[train_mask]
        X_val_dom_input = X_val_s[val_mask]
    else: # handcrafted
        X_train_dom_input = X_train_h_scaled[train_mask]
        X_val_dom_input = X_val_h_scaled[val_mask]

    y_train_dom_m = y_train_m[train_mask] # Binary maturity labels
    y_val_dom_m = y_val_m[val_mask]

    # Create Tensors and Datasets
    tensor_train_in = torch.tensor(X_train_dom_input, dtype=torch.float32)
    tensor_val_in = torch.tensor(X_val_dom_input, dtype=torch.float32)
    if FEATURE_TYPE == 'cnn': # Add channel dim for CNN extractor
        tensor_train_in = tensor_train_in.unsqueeze(1)
        tensor_val_in = tensor_val_in.unsqueeze(1)

    tensor_train_m = torch.tensor(y_train_dom_m, dtype=torch.long)
    tensor_val_m = torch.tensor(y_val_dom_m, dtype=torch.long)

    # Get features from the frozen Stage 1 extractor
    with torch.no_grad():
        features_train = model_s1(tensor_train_in.to(device)).cpu()
        features_val = model_s1(tensor_val_in.to(device)).cpu()

    train_dataset_s2 = TensorDataset(features_train, tensor_train_m)
    val_dataset_s2 = TensorDataset(features_val, tensor_val_m)

    train_loader_s2 = DataLoader(train_dataset_s2, batch_size=BATCH_SIZE, shuffle=True)
    val_loader_s2 = DataLoader(val_dataset_s2, batch_size=BATCH_SIZE * 2, shuffle=False)

    print(f"  Domain {domain_name} - Train samples: {len(train_dataset_s2)}, Val samples: {len(val_dataset_s2)}")

    # Instantiate Stage 2 Classifier
    model_s2 = MaturityClassifier(
        input_dim=EMBEDDING_DIM, # Input is the embedding from Stage 1
        hidden_dim=32,          # Can be tuned
        num_classes=2           # Ripe vs Rotten
    ).to(device)

    optimizer_s2 = optim.AdamW(model_s2.parameters(), lr=LEARNING_RATE_S2, weight_decay=WEIGHT_DECAY)
    scheduler_s2 = optim.lr_scheduler.ReduceLROnPlateau(optimizer_s2, 'min', patience=EARLY_STOPPING_PATIENCE_S2 // 2, factor=0.2, verbose=True)
    criterion_s2 = nn.CrossEntropyLoss()
    model_save_path = S2_MODEL_SAVE_TEMPLATE.format(domain_name)
    early_stopping_s2 = EarlyStopping(patience=EARLY_STOPPING_PATIENCE_S2, verbose=True, path=model_save_path)

    history_s2 = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    # --- Stage 2 Domain-Specific Training Loop ---
    for epoch in range(EPOCHS_S2):
        model_s2.train()
        running_loss = 0.0
        running_acc = 0.0
        num_batches = len(train_loader_s2)

        for features, labels in train_loader_s2:
            features, labels = features.to(device), labels.to(device)
            optimizer_s2.zero_grad()
            logits = model_s2(features)
            loss = criterion_s2(logits, labels)
            loss.backward()
            optimizer_s2.step()

            running_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            running_acc += (preds == labels).float().mean().item()

        epoch_train_loss = running_loss / num_batches
        epoch_train_acc = running_acc / num_batches
        history_s2['train_loss'].append(epoch_train_loss)
        history_s2['train_acc'].append(epoch_train_acc)

        # --- Stage 2 Domain-Specific Validation ---
        model_s2.eval()
        val_loss = 0.0
        val_acc = 0.0
        num_val_batches = len(val_loader_s2)
        with torch.no_grad():
            for features, labels in val_loader_s2:
                features, labels = features.to(device), labels.to(device)
                logits = model_s2(features)
                loss = criterion_s2(logits, labels)
                val_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                val_acc += (preds == labels).float().mean().item()

        epoch_val_loss = val_loss / num_val_batches
        epoch_val_acc = val_acc / num_val_batches
        history_s2['val_loss'].append(epoch_val_loss)
        history_s2['val_acc'].append(epoch_val_acc)

        print(f"  Epoch {epoch+1}/{EPOCHS_S2} | Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.4f} | Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.4f}")

        scheduler_s2.step(epoch_val_loss)
        early_stopping_s2(epoch_val_loss, model_s2)
        if early_stopping_s2.early_stop:
            print(f"  Early stopping triggered for domain {domain_name}.")
            break

    # Load best model for this domain
    print(f"  Loading best Stage 2 model for {domain_name} from {model_save_path}")
    # model_s2.load_state_dict(torch.load(model_save_path)) # Already loaded by EarlyStopping best_wts

    stage2_histories[domain_name] = history_s2
    plot_training_history(history_s2,
                          title=f'Stage 2 Maturity Classifier Training ({domain_name})',
                          save_path=S2_HISTORY_PLOT_TEMPLATE.format(domain_name))


s2_train_time = time.time() - start_time_s2
print(f"--- Stage 2 Training Completed in {s2_train_time:.2f} seconds ---")

print("\n--- Training Pipeline Finished ---")
total_time = time.time() - start_time
print(f"Total execution time: {total_time:.2f} seconds")

# --- Optional: Final Evaluation on Test Set ---
# It's better practice to do this in a separate test script,
# but we can add a quick check here.
print("\n--- Performing Final Evaluation on Test Set ---")
from test_pipeline import evaluate_on_test_set # Assuming test_pipeline.py exists

# Configuration for testing (matching training)
test_config = {
    "base_save_dir": BASE_SAVE_DIR,
    "feature_type": FEATURE_TYPE,
    "embedding_dim": EMBEDDING_DIM,
    "num_domains": NUM_DOMAINS,
    "domain_mapping": DOMAIN_MAPPING,
    "domain_reverse_mapping": DOMAIN_REVERSE_MAPPING,
    "expected_signal_length": EXPECTED_SIGNAL_LENGTH, # Needed if using CNN features
    "use_saved_test_data": True, # Use the test split saved during training
    "test_data_path": TEST_DATA_SAVE_PATH,
    "new_test_data_folders": None, # Not loading new data here
    "scaler_path": SCALER_SAVE_PATH,
    "s1_model_path": S1_MODEL_SAVE_PATH,
    "s1_prototypes_path": S1_PROTOTYPES_SAVE_PATH,
    "s2_model_template": S2_MODEL_SAVE_TEMPLATE,
    "device": device
}

try:
    evaluate_on_test_set(test_config, save_plots=True)
except NameError:
    print("\nSkipping final test set evaluation within training script.")
    print("Run test_pipeline.py separately for detailed testing.")
except Exception as e:
    print(f"\nError during final test set evaluation: {e}")