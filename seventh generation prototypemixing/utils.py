# utils.py
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl # For font settings
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.manifold import TSNE
import os
import copy
import config

# --- Set Matplotlib defaults for publication quality ---
# You might need to install specific fonts or adjust paths depending on your system
# plt.rcParams['font.family'] = 'serif' # Or 'sans-serif'
# plt.rcParams['font.serif'] = 'Times New Roman' # Or Arial, etc.
# plt.rcParams['mathtext.fontset'] = 'stix' # For math text if needed
plt.rcParams['axes.labelsize'] = config.PLOT_FONT_SIZE_LABELS if hasattr(config, 'PLOT_FONT_SIZE_LABELS') else 12
plt.rcParams['xtick.labelsize'] = config.PLOT_FONT_SIZE_TICKS if hasattr(config, 'PLOT_FONT_SIZE_TICKS') else 10
plt.rcParams['ytick.labelsize'] = config.PLOT_FONT_SIZE_TICKS if hasattr(config, 'PLOT_FONT_SIZE_TICKS') else 10
plt.rcParams['legend.fontsize'] = config.PLOT_FONT_SIZE_LEGEND if hasattr(config, 'PLOT_FONT_SIZE_LEGEND') else 10
plt.rcParams['figure.titlesize'] = config.PLOT_FONT_SIZE_TITLE if hasattr(config, 'PLOT_FONT_SIZE_TITLE') else 14
plt.rcParams['axes.titlesize'] = config.PLOT_FONT_SIZE_TITLE if hasattr(config, 'PLOT_FONT_SIZE_TITLE') else 14
plt.rcParams['figure.dpi'] = config.PLOT_DPI if hasattr(config, 'PLOT_DPI') else 150 # Higher DPI for saving
plt.rcParams['savefig.dpi'] = config.PLOT_DPI if hasattr(config, 'PLOT_DPI') else 300
plt.rcParams['savefig.format'] = 'png' # Or 'pdf', 'svg'
plt.rcParams['savefig.bbox'] = 'tight' # Adjust bounding box
plt.rcParams['lines.linewidth'] = 1.5
plt.rcParams['lines.markersize'] = 4

# --- EarlyStopping (No changes needed) ---
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
        self.best_model_wts = copy.deepcopy(model.state_dict())
        torch.save(self.best_model_wts, self.path)
        self.val_loss_min = val_loss


# --- Prototypical Network Functions (No changes needed) ---
def calculate_prototypes(features, labels, num_classes):
    """Calculates class prototypes (mean feature vector) from features."""
    device = features.device
    embedding_dim = features.size(1)
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
                 counts[c_int] = class_features.shape[0]
    return prototypes

def prototypical_loss(query_features, prototypes, query_labels, distance='cosine', temperature=1.0, return_preds=False):
    """
    Calculates the Prototypical Network loss and accuracy.
    Optionally returns predictions based on closest prototype.
    """
    device = query_features.device
    prototypes = prototypes.to(device)
    query_labels = query_labels.to(device)

    if distance == 'euclidean':
        prototypes = torch.nan_to_num(prototypes)
        dists = torch.cdist(query_features, prototypes) # (batch_size, num_classes)
        logits = -dists / temperature # Maximize similarity = Minimize distance
    elif distance == 'cosine':
        query_norm = F.normalize(query_features, p=2, dim=1)
        proto_norm = F.normalize(prototypes, p=2, dim=1)
        proto_norm = torch.nan_to_num(proto_norm)
        logits = torch.mm(query_norm, proto_norm.t()) * temperature # Cosine similarity directly serves as logits
    else:
        raise ValueError(f"Unsupported distance metric: {distance}")

    # --- Prediction ---
    # Predicted class is the one with the highest logit (max similarity / min distance)
    preds = torch.argmax(logits, dim=1)

    # --- Loss Calculation (if labels provided) ---
    loss = torch.tensor(0.0, device=device) # Default loss if no labels
    acc = torch.tensor(0.0, device=device) # Default accuracy if no labels
    if query_labels is not None and len(query_labels) > 0:
        # Handle potential missing labels if query_labels don't cover 0..N-1
        max_label = query_labels.max().item()
        if max_label >= logits.shape[1]:
            print(f"Error: Max query label ({max_label}) is out of bounds for logits shape ({logits.shape}). Check prototype calculation or labels.")
            # Return high loss, zero acc, and potentially handle preds based on context
            dummy_preds = torch.zeros_like(query_labels)
            if return_preds:
                return torch.tensor(10.0, device=device, requires_grad=True), torch.tensor(0.0, device=device), dummy_preds
            else:
                return torch.tensor(10.0, device=device, requires_grad=True), torch.tensor(0.0, device=device)
        else:
            # Calculate Cross-Entropy Loss
            loss = F.cross_entropy(logits, query_labels.long())
            # Calculate accuracy for monitoring
            acc = (preds == query_labels).float().mean()

    if return_preds:
        return loss, acc, preds
    else:
        return loss, acc

# --- Feature Mixing Functions (No changes needed) ---
def mix_features(features, labels, num_classes, num_mixes_per_class=1, alpha=0.2):
    device = features.device
    mixed_features = []
    mixed_labels = [] # Using soft labels
    if len(features) < 2: return torch.empty(0, features.shape[1], device=device), torch.empty(0, num_classes, device=device)
    unique_labels_present = torch.unique(labels)
    if len(unique_labels_present) < 2: return torch.empty(0, features.shape[1], device=device), torch.empty(0, num_classes, device=device)
    if unique_labels_present.max() >= num_classes or unique_labels_present.min() < 0:
        print(f"Warning: Labels provided to mix_features ({unique_labels_present}) outside [0, {num_classes-1}]. Skipping.")
        return torch.empty(0, features.shape[1], device=device), torch.empty(0, num_classes, device=device)

    for c1_tensor in unique_labels_present:
        c1 = c1_tensor.long()
        indices_c1 = torch.where(labels == c1)[0]
        if len(indices_c1) == 0: continue
        other_labels = unique_labels_present[unique_labels_present != c1]
        if len(other_labels) == 0: continue

        for _ in range(num_mixes_per_class):
            idx1 = indices_c1[torch.randint(len(indices_c1), (1,), device=device).item()]
            c2_tensor = other_labels[torch.randint(len(other_labels), (1,), device=device).item()]
            c2 = c2_tensor.long()
            indices_c2 = torch.where(labels == c2)[0]
            if len(indices_c2) == 0: continue
            idx2 = indices_c2[torch.randint(len(indices_c2), (1,), device=device).item()]
            lam = np.random.beta(alpha, alpha) if alpha > 0 else 0.5
            lam = torch.tensor(lam, device=device, dtype=features.dtype)
            mixed_feat = lam * features[idx1] + (1 - lam) * features[idx2]
            mixed_features.append(mixed_feat)
            soft_label = torch.zeros(num_classes, device=device, dtype=features.dtype)
            soft_label[c1] = lam
            soft_label[c2] = 1 - lam
            mixed_labels.append(soft_label)

    if not mixed_features: return torch.empty(0, features.shape[1], device=device), torch.empty(0, num_classes, device=device)
    return torch.stack(mixed_features), torch.stack(mixed_labels)

def mixed_feature_loss(mixed_features, mixed_soft_labels, prototypes, distance='cosine', temperature=1.0):
    if len(mixed_features) == 0: return torch.tensor(0.0, device=mixed_features.device)
    device = mixed_features.device
    prototypes = prototypes.to(device)

    if distance == 'euclidean':
        prototypes = torch.nan_to_num(prototypes)
        dists = torch.cdist(mixed_features, prototypes)
        logits = -dists / temperature
    elif distance == 'cosine':
        mixed_norm = F.normalize(mixed_features, p=2, dim=1)
        proto_norm = F.normalize(prototypes, p=2, dim=1); proto_norm = torch.nan_to_num(proto_norm)
        logits = torch.mm(mixed_norm, proto_norm.t()) * temperature
    else: raise ValueError(f"Unsupported distance metric: {distance}")

    log_pred_probs = F.log_softmax(logits, dim=1)
    loss = F.kl_div(log_pred_probs, mixed_soft_labels, reduction='batchmean', log_target=False)
    if torch.isnan(loss):
         print("Warning: NaN detected in mixed_feature_loss."); return torch.tensor(0.0, device=device)
    return loss


# --- Visualization Functions (Enhanced) ---
def plot_training_history(history, title='Training History', save_path=None, figsize=(10, 4.5)):
    """Plots training and validation loss and accuracy with enhanced style."""
    if not history or 'train_loss' not in history or not history['train_loss']:
        print("Cannot plot training history: History data is missing or empty.")
        return

    epochs = range(1, len(history['train_loss']) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle(title, fontsize=plt.rcParams['figure.titlesize'] * 1.1) # Slightly larger overall title

    # Plot Loss
    ax1.plot(epochs, history.get('train_loss', []), 'o-', color='royalblue', label='Training loss')
    ax1.plot(epochs, history.get('val_loss', []), 's--', color='orangered', label='Validation loss') # Changed style
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss over Epochs')
    ax1.legend()
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.spines['top'].set_visible(False) # Cleaner look
    ax1.spines['right'].set_visible(False)


    # Plot Accuracy
    ax2.plot(epochs, history.get('train_acc', []), 'o-', color='royalblue', label='Training accuracy')
    ax2.plot(epochs, history.get('val_acc', []), 's--', color='orangered', label='Validation accuracy') # Changed style
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy over Epochs')
    # Set y-axis limits for accuracy, ensure 0 and 1 (or 1.05) are visible
    min_acc = min(0, min(history.get('val_acc', [0])) - 0.05) if history.get('val_acc') else 0
    max_acc = max(1.0, max(history.get('val_acc', [1])) + 0.05) if history.get('val_acc') else 1.05
    ax2.set_ylim(min_acc, max_acc)
    ax2.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1.0)) # Show as percentage

    ax2.legend()
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout slightly for main title
    if save_path:
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            # Use parameters from config if available, otherwise defaults in rcParams
            save_dpi = config.PLOT_DPI if hasattr(config, 'PLOT_DPI') else plt.rcParams['savefig.dpi']
            plt.savefig(save_path, dpi=save_dpi, bbox_inches='tight')
            print(f"Training history plot saved to {save_path}")
        except Exception as e:
            print(f"Error saving training history plot to {save_path}: {e}")
    plt.close() # Close plot to free memory

def plot_confusion_matrix_heatmap(y_true, y_pred, class_names, title='Confusion Matrix', save_path=None, cmap='viridis', figsize=None, annot_fontsize=10):
    """Plots a confusion matrix using seaborn heatmap with enhanced style."""
    if len(y_true) == 0 or len(y_pred) == 0:
         print(f"Cannot plot confusion matrix '{title}': No data provided.")
         return
    if len(y_true) != len(y_pred):
         print(f"Error plotting confusion matrix '{title}': y_true ({len(y_true)}) and y_pred ({len(y_pred)}) have different lengths.")
         return

    present_labels_true = np.unique(y_true).astype(int)
    present_labels_pred = np.unique(y_pred).astype(int)
    all_present_labels = np.unique(np.concatenate((present_labels_true, present_labels_pred)))

    # Ensure labels used for CM cover all present labels and expected classes based on class_names
    expected_labels_indices = np.arange(len(class_names)).astype(int)
    labels_to_include = np.unique(np.concatenate((all_present_labels, expected_labels_indices))).astype(int)

    # Create tick labels based on the indices actually used in the CM plot
    tick_labels = [class_names[i] if i < len(class_names) else f"Idx_{i}" for i in labels_to_include]

    try:
        cm = confusion_matrix(y_true, y_pred, labels=labels_to_include)
        # Automatic figsize based on number of classes
        if figsize is None:
            figsize = (max(5, len(tick_labels)*0.6 + 1), max(4, len(tick_labels)*0.5 + 1))

        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt="d", cmap=cmap,
                    xticklabels=tick_labels, yticklabels=tick_labels,
                    annot_kws={"size": annot_fontsize})
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45, ha='right') # Rotate labels if they overlap
        plt.yticks(rotation=0)
        plt.tight_layout()
        if save_path:
             os.makedirs(os.path.dirname(save_path), exist_ok=True)
             save_dpi = config.PLOT_DPI if hasattr(config, 'PLOT_DPI') else plt.rcParams['savefig.dpi']
             plt.savefig(save_path, dpi=save_dpi, bbox_inches='tight')
             print(f"Confusion matrix saved to {save_path}")
        plt.close()
    except Exception as e:
         print(f"Error generating confusion matrix '{title}': {e}")
         print(f"  y_true unique: {np.unique(y_true)}")
         print(f"  y_pred unique: {np.unique(y_pred)}")
         print(f"  Labels used for CM: {labels_to_include}")
         print(f"  Class names provided: {class_names}")

def plot_tsne(features, labels, class_names_map, title='t-SNE Visualization', save_path=None, figsize=(9, 7), perplexity=30.0, n_iter=300, marker_size=None, alpha=0.7, cmap='viridis'):
    """Plots t-SNE visualization of features with enhanced style."""
    if features is None or len(features) == 0:
        print(f"Cannot plot t-SNE '{title}': No features provided.")
        return
    if labels is None or len(labels) != len(features):
        print(f"Error plotting t-SNE '{title}': Mismatch features ({len(features)}) vs labels ({len(labels) if labels is not None else 'None'}).")
        return

    n_samples = len(features)
    if n_samples <= 1:
        print(f"Not enough samples ({n_samples}) for t-SNE plot '{title}'.")
        return

    # Use marker size from config or default
    marker_size = marker_size if marker_size is not None else (config.PLOT_MARKER_SIZE if hasattr(config, 'PLOT_MARKER_SIZE') else 20)

    # Adjust perplexity if default is too high for sample size
    perplexity_adjusted = min(perplexity, float(n_samples - 1.1)) # Must be < n_samples
    if perplexity_adjusted <= 1: perplexity_adjusted = max(5.0, float(n_samples - 1.1)) # Ensure reasonable min
    if perplexity_adjusted != perplexity:
        print(f"Adjusted t-SNE perplexity from {perplexity} to {perplexity_adjusted:.1f} due to sample size {n_samples}.")

    print(f"Running t-SNE for '{title}' (perplexity={perplexity_adjusted:.1f}, n_iter={n_iter})... on {n_samples} samples.")
    tsne = TSNE(n_components=2, random_state=config.SEED if hasattr(config, 'SEED') else 42,
                perplexity=perplexity_adjusted, n_iter=n_iter, init='pca', learning_rate='auto',
                n_jobs=-1) # Use multiple cores

    try:
        # Ensure features are float32 numpy array on CPU
        if isinstance(features, torch.Tensor): features_np = features.detach().cpu().numpy().astype(np.float32)
        elif isinstance(features, np.ndarray): features_np = features.astype(np.float32)
        else: features_np = np.array(features).astype(np.float32)
        if np.isnan(features_np).any() or np.isinf(features_np).any():
             print(f"Warning: Features for t-SNE '{title}' contain NaN/Inf. Replacing with 0.")
             features_np = np.nan_to_num(features_np)
        tsne_results = tsne.fit_transform(features_np)
    except ValueError as ve: # Handle PCA fallback for small perplexity cases
         print(f"t-SNE failed for '{title}': {ve}. Attempting PCA fallback...")
         try:
             from sklearn.decomposition import PCA
             pca = PCA(n_components=2)
             tsne_results = pca.fit_transform(features_np)
             title = title.replace("t-SNE", "PCA Fallback")
         except Exception as pca_e: print(f"PCA fallback failed: {pca_e}. Skipping plot."); return
    except Exception as e: print(f"Unexpected error during t-SNE/PCA: {e}. Skipping plot."); return

    plt.figure(figsize=figsize)
    unique_labels = np.unique(labels)
    colors = plt.get_cmap(cmap, len(unique_labels)) # Get distinct colors from cmap
    label_to_color = {label: colors(i) for i, label in enumerate(unique_labels)}

    plotted_labels = set()
    for i in range(n_samples):
        label = labels[i]
        color = label_to_color.get(label, 'gray') # Fallback color
        label_name = class_names_map.get(label, f"Unknown: {label}") # Get readable name

        if label not in plotted_labels:
             plt.scatter(tsne_results[i, 0], tsne_results[i, 1], color=color, label=label_name, alpha=alpha, s=marker_size, edgecolors='w', linewidth=0.3) # Add faint edge
             plotted_labels.add(label)
        else:
             plt.scatter(tsne_results[i, 0], tsne_results[i, 1], color=color, alpha=alpha, s=marker_size, edgecolors='w', linewidth=0.3)

    plt.title(title)
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    # Place legend smartly
    if len(unique_labels) <= 10: # Internal legend for fewer classes
        plt.legend(title="Classes", loc='best')
        plt.tight_layout()
    else: # External legend for many classes
        plt.legend(title="Classes", bbox_to_anchor=(1.02, 1), loc='upper left')
        plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout for external legend

    plt.grid(True, linestyle=':', alpha=0.5)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


    if save_path:
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            save_dpi = config.PLOT_DPI if hasattr(config, 'PLOT_DPI') else plt.rcParams['savefig.dpi']
            plt.savefig(save_path, dpi=save_dpi, bbox_inches='tight')
            print(f"t-SNE/PCA plot saved to {save_path}")
        except Exception as e:
            print(f"Error saving t-SNE/PCA plot to {save_path}: {e}")
    plt.close()