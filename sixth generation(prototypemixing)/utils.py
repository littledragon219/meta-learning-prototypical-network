# utils.py
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.manifold import TSNE
import os
import copy
import config
# --- EarlyStopping (keep as before) ---
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pth', trace_func=print):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        self.best_model_wts = None

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        # Ensure model is on CPU before saving state_dict to avoid GPU tensors in file
        # model_cpu = copy.deepcopy(model).to('cpu')
        # self.best_model_wts = model_cpu.state_dict()
        # Use model's current state_dict directly, loading handles map_location
        self.best_model_wts = copy.deepcopy(model.state_dict())

        torch.save(self.best_model_wts, self.path)
        self.val_loss_min = val_loss

# --- Prototypical Network Functions ---
def calculate_prototypes(features, labels, num_classes):
    """Calculates class prototypes (mean feature vector) from features."""
    # Ensure features and labels are on the same device, preferably CPU for calculation stability?
    # Or keep on config.DEVICE if sufficient memory. Let's keep on feature device.
    device = features.device
    embedding_dim = features.size(1)

    # Initialize prototypes with zeros ON THE SAME DEVICE
    prototypes = torch.zeros(num_classes, embedding_dim, device=device)
    counts = torch.zeros(num_classes, device=device)

    unique_labels = torch.unique(labels)

    for c in unique_labels:
        c_int = c.item() # Convert tensor label to integer index
        if 0 <= c_int < num_classes: # Ensure label is within valid range
            class_mask = (labels == c)
            class_features = features[class_mask]
            if class_features.shape[0] > 0:
                 prototypes[c_int] = class_features.mean(dim=0)
                 counts[c_int] = class_features.shape[0] # Keep track of counts (optional)
            # else: prototype remains zero if no samples for this class index
        # else: print(f"Warning: Label {c_int} out of range [0, {num_classes-1}]. Skipping prototype.")

    # Optional: Handle classes with zero samples? For now, they have zero prototypes.
    # Could replace zero prototypes with overall mean or random vector if needed.

    return prototypes # Shape: (num_classes, embedding_dim)


def prototypical_loss(query_features, prototypes, query_labels, distance='cosine', temperature=1.0, return_preds=False):
    """
    Calculates the Prototypical Network loss and accuracy.
    Optionally returns predictions.
    """
    device = query_features.device
    # Ensure prototypes are on the same device as features
    prototypes = prototypes.to(device)
    query_labels = query_labels.to(device)


    if distance == 'euclidean':
        # Ensure prototypes don't have NaNs/Infs if some classes had no samples
        prototypes = torch.nan_to_num(prototypes)
        dists = torch.cdist(query_features, prototypes) # (batch_size, num_classes)
        logits = -dists / temperature # Scale negative distance
    elif distance == 'cosine':
        # Normalize features and prototypes
        query_norm = F.normalize(query_features, p=2, dim=1)
        proto_norm = F.normalize(prototypes, p=2, dim=1)
        # Handle potential zero vectors in prototypes (if a class had no samples)
        # Replace NaN resulting from normalizing zero vectors with a very small value or zero similarity
        proto_norm = torch.nan_to_num(proto_norm)

        # Calculate cosine similarity (dot product of normalized vectors)
        logits = torch.mm(query_norm, proto_norm.t()) * temperature
    else:
        raise ValueError(f"Unsupported distance metric: {distance}")

    # Handle potential missing labels if query_labels don't cover 0..N-1
    # Check if max label index is valid for the logits shape
    max_label = -1
    if len(query_labels) > 0 :
         max_label = query_labels.max().item()

    if max_label >= logits.shape[1]:
        print(f"Error: Max query label ({max_label}) is out of bounds for logits shape ({logits.shape}). Check prototype calculation or labels.")
        # Return high loss, zero acc, and potentially handle preds based on context
        dummy_preds = torch.zeros_like(query_labels)
        if return_preds:
            return torch.tensor(10.0, device=device, requires_grad=True), torch.tensor(0.0, device=device), dummy_preds
        else:
            return torch.tensor(10.0, device=device, requires_grad=True), torch.tensor(0.0, device=device)


    # Calculate Cross-Entropy Loss
    # Ensure labels are long type
    loss = F.cross_entropy(logits, query_labels.long())

    # Calculate accuracy for monitoring
    preds = torch.argmax(logits, dim=1)
    # Ensure comparison happens on the same device
    acc = (preds == query_labels).float().mean()

    if return_preds:
        return loss, acc, preds
    else:
        return loss, acc

# --- Feature Mixing Function (keep as before) ---
def mix_features(features, labels, num_classes, num_mixes_per_class=1, alpha=0.2):
    """
    Performs MixUp-style augmentation directly on feature embeddings.
    Mixes features from different classes.
    """
    device = features.device
    mixed_features = []
    mixed_labels = [] # Using soft labels

    if len(features) < 2: # Need at least two samples to mix
        return torch.empty(0, features.shape[1], device=device), torch.empty(0, num_classes, device=device)

    unique_labels_present = torch.unique(labels)
    if len(unique_labels_present) < 2: # Need at least two different classes to mix
        return torch.empty(0, features.shape[1], device=device), torch.empty(0, num_classes, device=device)

    # Ensure labels used for indexing are within range [0, num_classes-1]
    if unique_labels_present.max() >= num_classes or unique_labels_present.min() < 0:
        print(f"Warning: Labels provided to mix_features ({unique_labels_present}) are outside the expected range [0, {num_classes-1}]. Skipping mixing.")
        return torch.empty(0, features.shape[1], device=device), torch.empty(0, num_classes, device=device)


    for c1_tensor in unique_labels_present:
        c1 = c1_tensor.long() # Ensure integer type for indexing
        indices_c1 = torch.where(labels == c1)[0]
        if len(indices_c1) == 0: continue

        # Find other classes present
        other_labels = unique_labels_present[unique_labels_present != c1]
        if len(other_labels) == 0: continue


        for _ in range(num_mixes_per_class):
            # Pick a random sample from class c1
            idx1 = indices_c1[torch.randint(len(indices_c1), (1,), device=device).item()]

            # Pick a random other class c2 and a random sample from it
            c2_tensor = other_labels[torch.randint(len(other_labels), (1,), device=device).item()]
            c2 = c2_tensor.long() # Ensure integer type
            indices_c2 = torch.where(labels == c2)[0]
            if len(indices_c2) == 0: continue # Should not happen based on logic above, but safe check
            idx2 = indices_c2[torch.randint(len(indices_c2), (1,), device=device).item()]

            # Sample mixing ratio lambda from Beta distribution
            lam = np.random.beta(alpha, alpha) if alpha > 0 else 0.5
            lam = torch.tensor(lam, device=device, dtype=features.dtype) # Ensure lambda is tensor on correct device/dtype

            # Mix features
            mixed_feat = lam * features[idx1] + (1 - lam) * features[idx2]
            mixed_features.append(mixed_feat)

            # Create soft labels
            soft_label = torch.zeros(num_classes, device=device, dtype=features.dtype)
            soft_label[c1] = lam
            soft_label[c2] = 1 - lam
            mixed_labels.append(soft_label)

    if not mixed_features:
        return torch.empty(0, features.shape[1], device=device), torch.empty(0, num_classes, device=device)

    return torch.stack(mixed_features), torch.stack(mixed_labels)


def mixed_feature_loss(mixed_features, mixed_soft_labels, prototypes, distance='cosine', temperature=1.0):
    """
    Calculates loss for mixed features against prototypes.
    Uses KL divergence between predicted distribution (softmax over logits) and soft labels.
    """
    if len(mixed_features) == 0:
        return torch.tensor(0.0, device=mixed_features.device)

    device = mixed_features.device
    prototypes = prototypes.to(device) # Ensure prototypes are on correct device

    if distance == 'euclidean':
        prototypes = torch.nan_to_num(prototypes) # Handle potential NaNs
        dists = torch.cdist(mixed_features, prototypes) # (N_mix, num_classes)
        logits = -dists / temperature
    elif distance == 'cosine':
        mixed_norm = F.normalize(mixed_features, p=2, dim=1)
        proto_norm = F.normalize(prototypes, p=2, dim=1)
        proto_norm = torch.nan_to_num(proto_norm) # Handle zero prototypes
        logits = torch.mm(mixed_norm, proto_norm.t()) * temperature
    else:
        raise ValueError(f"Unsupported distance metric: {distance}")

    # Predicted probability distribution using log_softmax for KLDivLoss stability
    log_pred_probs = F.log_softmax(logits, dim=1)

    # KL divergence loss expects log-probabilities as input (log_pred_probs)
    # and probabilities as target (mixed_soft_labels)
    # Ensure mixed_soft_labels sums to 1 for valid probability distribution
    # Note: mixed_soft_labels generated by mix_features should already sum to 1.
    loss = F.kl_div(log_pred_probs, mixed_soft_labels, reduction='batchmean', log_target=False) # log_target=False since target is probabilities

    # Handle potential NaN loss if inputs cause issues
    if torch.isnan(loss):
         print("Warning: NaN detected in mixed_feature_loss. Check inputs/logits.")
         # print("Logits:", logits)
         # print("Log Pred Probs:", log_pred_probs)
         # print("Soft Labels:", mixed_soft_labels)
         return torch.tensor(0.0, device=device) # Return 0 loss to avoid crashing training


    return loss


# --- Visualization Functions (Optimized for CPU/speed) ---
def plot_training_history(history, title='Training History', save_path=None):
    """Plots training and validation loss and accuracy."""
    if not history or 'train_loss' not in history or not history['train_loss']:
        print("Cannot plot training history: History data is missing or empty.")
        return

    epochs = range(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(12, 5))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history.get('train_loss', []), 'bo-', label='Training loss')
    plt.plot(epochs, history.get('val_loss', []), 'ro-', label='Validation loss')
    plt.title(f'{title} - Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history.get('train_acc', []), 'bo-', label='Training accuracy')
    plt.plot(epochs, history.get('val_acc', []), 'ro-', label='Validation accuracy')
    plt.title(f'{title} - Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    # Set y-axis limits for accuracy if desired (e.g., 0 to 1 or slightly more)
    if history.get('val_acc'): plt.ylim(min(0, min(history['val_acc'])-0.1), max(1.05, max(history['val_acc'])+0.05 ))

    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    if save_path:
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            print(f"Training history plot saved to {save_path}")
        except Exception as e:
            print(f"Error saving training history plot to {save_path}: {e}")
    # plt.show() # Avoid showing plots during automated runs
    plt.close() # Close plot to free memory


def plot_confusion_matrix_heatmap(y_true, y_pred, class_names, title='Confusion Matrix', save_path=None):
    """Plots a confusion matrix using seaborn heatmap."""
    if len(y_true) == 0 or len(y_pred) == 0:
         print(f"Cannot plot confusion matrix '{title}': No data provided.")
         return
    if len(y_true) != len(y_pred):
         print(f"Error plotting confusion matrix '{title}': y_true ({len(y_true)}) and y_pred ({len(y_pred)}) have different lengths.")
         return

    # Ensure labels cover all classes present in y_true or y_pred + expected classes
    present_labels = np.unique(np.concatenate((y_true, y_pred)))
    all_labels_indices = np.arange(len(class_names)) # Indices corresponding to class_names
    # Combine present labels and expected labels, ensuring they are integers
    labels_to_include = np.unique(np.concatenate((present_labels.astype(int), all_labels_indices))).astype(int)


    try:
        cm = confusion_matrix(y_true, y_pred, labels=labels_to_include)
        # Ensure class_names corresponds correctly to labels_to_include if they differ
        # If labels_to_include has more indices than class_names, need placeholder names
        tick_labels = [class_names[i] if i < len(class_names) else f"Idx_{i}" for i in labels_to_include]


        plt.figure(figsize=(max(6, len(tick_labels)*0.8), max(5, len(tick_labels)*0.6))) # Adjust size
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=tick_labels, yticklabels=tick_labels)
        plt.title(title)
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        if save_path:
             os.makedirs(os.path.dirname(save_path), exist_ok=True)
             plt.savefig(save_path)
             print(f"Confusion matrix saved to {save_path}")
        # plt.show()
        plt.close()
    except Exception as e:
         print(f"Error generating confusion matrix '{title}': {e}")
         # This can happen if labels are inconsistent or indexing fails
         print(f"  y_true unique: {np.unique(y_true)}")
         print(f"  y_pred unique: {np.unique(y_pred)}")
         print(f"  Labels used: {labels_to_include}")
         print(f"  Class names provided: {class_names}")


def plot_tsne(features, labels, class_names_map, title='t-SNE Visualization', save_path=None):
    """Plots t-SNE visualization of features. Optimized for speed."""
    if features is None or len(features) == 0:
        print(f"Cannot plot t-SNE '{title}': No features provided.")
        return
    if labels is None or len(labels) != len(features):
        print(f"Error plotting t-SNE '{title}': Mismatch between features ({len(features)}) and labels ({len(labels) if labels is not None else 'None'}).")
        return

    n_samples = len(features)
    if n_samples <= 1:
        print(f"Not enough samples ({n_samples}) for t-SNE plot '{title}'.")
        return

    # Adjust perplexity based on sample size
    perplexity = min(30.0, max(5.0, float(n_samples / 5))) # Scale perplexity, min 5
    perplexity = min(perplexity, float(n_samples - 1)) # Perplexity must be < n_samples
    if perplexity <= 0: perplexity = 5.0 # Final fallback

    # Reduce iterations for faster plotting, especially on CPU
    n_iter = 250 # Reduced from 300 or 1000

    print(f"Running t-SNE for '{title}' (perplexity={perplexity:.1f}, n_iter={n_iter})... on {n_samples} samples.")
    tsne = TSNE(n_components=2, random_state=config.SEED if hasattr(config, 'SEED') else 42,
                perplexity=perplexity, n_iter=n_iter, init='pca', learning_rate='auto',
                n_jobs=-1) # Use multiple cores if possible

    try:
        # Ensure features are float32 numpy array on CPU
        if isinstance(features, torch.Tensor):
             features_np = features.detach().cpu().numpy().astype(np.float32)
        elif isinstance(features, np.ndarray):
             features_np = features.astype(np.float32)
        else:
             features_np = np.array(features).astype(np.float32)

        if np.isnan(features_np).any() or np.isinf(features_np).any():
             print(f"Warning: Features for t-SNE '{title}' contain NaN or Inf. Replacing with 0.")
             features_np = np.nan_to_num(features_np)


        tsne_results = tsne.fit_transform(features_np)
    except ValueError as ve:
         print(f"t-SNE failed for '{title}': {ve}")
         print("Attempting PCA fallback...")
         try:
             from sklearn.decomposition import PCA
             pca = PCA(n_components=2)
             tsne_results = pca.fit_transform(features_np)
             title = title.replace("t-SNE", "PCA Fallback")
         except Exception as pca_e:
             print(f"PCA fallback also failed for '{title}': {pca_e}. Skipping plot.")
             return
    except Exception as e:
         print(f"An unexpected error occurred during t-SNE/PCA for '{title}': {e}. Skipping plot.")
         return


    plt.figure(figsize=(10, 8))
    unique_labels = np.unique(labels)
    # Use a perceptually uniform colormap like 'viridis' or 'plasma'
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))
    label_to_color = {label: color for label, color in zip(unique_labels, colors)}

    # Plot points class by class for legend handling
    plotted_labels = set()
    for i in range(n_samples):
        label = labels[i]
        color = label_to_color.get(label, 'gray') # Fallback color for unexpected labels
        # Map integer label to string name using class_names_map
        label_name = class_names_map.get(label, f"Unknown: {label}")

        if label not in plotted_labels:
             plt.scatter(tsne_results[i, 0], tsne_results[i, 1], color=color, label=label_name, alpha=0.6, s=15) # Smaller points
             plotted_labels.add(label)
        else:
             plt.scatter(tsne_results[i, 0], tsne_results[i, 1], color=color, alpha=0.6, s=15)

    plt.title(title)
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    # Place legend outside plot for clarity if many classes
    plt.legend(title="Classes", bbox_to_anchor=(1.02, 1), loc='upper left', fontsize='small')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for external legend

    if save_path:
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight') # Use bbox_inches='tight' for external legend
            print(f"t-SNE/PCA plot saved to {save_path}")
        except Exception as e:
            print(f"Error saving t-SNE/PCA plot to {save_path}: {e}")

    # plt.show()
    plt.close() # Close plot to free memory