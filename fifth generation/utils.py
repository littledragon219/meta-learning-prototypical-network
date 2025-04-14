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
        # Ensure the directory exists
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        # Save model state dictionary
        self.best_model_wts = copy.deepcopy(model.state_dict())
        torch.save(self.best_model_wts, self.path)
        self.val_loss_min = val_loss

def calculate_prototypes(features, labels, num_classes):
    """Calculates class prototypes (mean feature vector) from support features."""
    prototypes = []
    device = features.device
    embedding_dim = features.size(1)
    for c in range(num_classes):
        class_features = features[labels == c]
        if len(class_features) > 0:
            prototype = class_features.mean(dim=0)
        else:
            # Handle case where a class might be missing in a batch/set
            prototype = torch.zeros(embedding_dim, device=device)
            # print(f"Warning: Class {c} has no samples for prototype calculation.")
        prototypes.append(prototype)
    return torch.stack(prototypes) # Shape: (num_classes, embedding_dim)

def prototypical_loss(query_features, prototypes, query_labels, distance='euclidean'):
    """Calculates the Prototypical Network loss."""
    # query_features: (batch_size, embedding_dim)
    # prototypes: (num_classes, embedding_dim)
    # query_labels: (batch_size,)

    if distance == 'euclidean':
        # Calculate squared Euclidean distances (more stable than sqrt)
        dists = torch.sum((query_features.unsqueeze(1) - prototypes.unsqueeze(0))**2, dim=2)
        # Use negative distance as logits (closer distance = higher probability)
        logits = -dists
    elif distance == 'cosine':
        # Calculate cosine similarity, scale it (optional but common)
        logits = F.cosine_similarity(query_features.unsqueeze(1), prototypes.unsqueeze(0), dim=-1) * 10.0 # Add temperature scaling
    else:
        raise ValueError("Unsupported distance metric")

    # Cross-entropy loss expects logits where higher values mean higher probability
    loss = F.cross_entropy(logits, query_labels)
    # Calculate accuracy for monitoring
    preds = torch.argmax(logits, dim=1)
    acc = (preds == query_labels).float().mean()

    return loss, acc

# --- Visualization Functions ---

def plot_training_history(history, title='Training History', save_path=None):
    """Plots training and validation loss and accuracy."""
    epochs = range(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(12, 5))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'bo-', label='Training loss')
    plt.plot(epochs, history['val_loss'], 'ro-', label='Validation loss')
    plt.title(f'{title} - Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], 'bo-', label='Training accuracy')
    plt.plot(epochs, history['val_acc'], 'ro-', label='Validation accuracy')
    plt.title(f'{title} - Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Training history plot saved to {save_path}")
    plt.show()


def plot_confusion_matrix_heatmap(y_true, y_pred, class_names, title='Confusion Matrix', save_path=None):
    """Plots a confusion matrix using seaborn heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    if save_path:
         os.makedirs(os.path.dirname(save_path), exist_ok=True)
         plt.savefig(save_path)
         print(f"Confusion matrix saved to {save_path}")
    plt.show()

def plot_tsne(features, labels, class_names_map, title='t-SNE Visualization', save_path=None):
    """Plots t-SNE visualization of features."""
    if len(features) <= 1:
        print("Not enough samples for t-SNE plot.")
        return
    perplexity = min(30.0, float(len(features) - 1)) # Adjust perplexity based on sample size
    if perplexity <= 0:
        print("Perplexity must be positive for t-SNE.")
        return

    print(f"Running t-SNE (perplexity={perplexity})...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, n_iter=500, init='pca')
    try:
        tsne_results = tsne.fit_transform(features)
    except Exception as e:
         print(f"t-SNE failed: {e}")
         # Fallback: Try PCA instead
         from sklearn.decomposition import PCA
         print("Falling back to PCA...")
         pca = PCA(n_components=2)
         tsne_results = pca.fit_transform(features)
         title = title.replace("t-SNE", "PCA")


    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels, cmap='viridis', alpha=0.7)
    plt.title(title)
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')

    # Create legend
    try:
        handles, current_labels = scatter.legend_elements()
        # Map numeric labels back to names using the provided dict
        legend_labels = [class_names_map.get(int(label.split('{')[-1].split('}')[0]), f"Unknown: {label}") for label in current_labels]
        plt.legend(handles=handles, labels=legend_labels, title="Classes")
    except Exception as e:
        print(f"Could not create legend: {e}")
        plt.colorbar(label='Class Label') # Fallback colorbar

    plt.grid(True)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"t-SNE plot saved to {save_path}")
    plt.show()