# utils.py
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import torch
from sklearn.metrics import f1_score, confusion_matrix
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
import numpy as np
import math
# Seed for reproducibility - good practice
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

def grl_lambda_schedule(epoch, total_epochs, alpha=10.0, beta=0.75, verbose=False):
    """
    Calculates the GRL lambda dynamically based on training progress.
    Lambda increases from 0 to 1 following sigmoid-like curve.

    Args:
        epoch (int): Current epoch number (starting from 0).
        total_epochs (int): Total number of training epochs.
        alpha (float): Controls the steepness of the transition.
        beta (float): Controls the point where transition happens (0 to 1).
                      Default 0.75 means lambda starts increasing significantly
                      around 75% of the way through training. (Incorrect intuition, actually related to sigmoid shift)
                      Let's adjust the formula for clearer control. See below.
        verbose (bool): Print the calculated lambda.

    Returns:
        float: The calculated lambda value for GRL (between 0 and 1).
    """
    # Alternative formula: Sigmoid curve centered around `center_epoch`
    # p = float(epoch) / total_epochs
    # lambda_p = (2. / (1. + math.exp(-alpha * p))) - 1

    # Formula often cited (e.g., in DANN paper):
    p = float(epoch) / total_epochs
    lambda_p = 2. / (1. + math.exp(-alpha * p)) - 1 # Ranges from 0 to 1 as p goes 0 to 1

    # Let's use a slightly adjusted version for better control potentially
    # Centered sigmoid: transition happens around center_epoch
    # center_epoch_ratio = 0.5 # e.g., transition centered at 50% epochs
    # lambda_p = 1. / (1. + math.exp(-alpha * (p - center_epoch_ratio)))

    # Let's stick to the common DANN formula for now:
    lambda_p = 2. / (1. + math.exp(-alpha * p)) - 1

    # Clamp value between 0 and 1 just in case
    lambda_p = max(0.0, min(1.0, lambda_p))

    if verbose and (epoch % (total_epochs // 10) == 0 or epoch == total_epochs -1) : # Print periodically
        print(f"Epoch {epoch+1}/{total_epochs}: GRL Lambda = {lambda_p:.4f}")

    return lambda_p

def robust_imputation(features, name="features"):
    """
    对输入的特征矩阵进行鲁棒的缺失值填充:
      - 对于某列全为 NaN，则替换为0，
      - 其余部分采用 SimpleImputer 处理。
    """
    num_nan_before = np.isnan(features).sum()
    if num_nan_before > 0:
        print(f"{name} - NaN数量（填充前）: {num_nan_before}")

        # 检查并处理完全是 NaN 的列
        col_is_nan = np.all(np.isnan(features), axis=0)
        if np.any(col_is_nan):
            print(f"{name} - 检测到 {np.sum(col_is_nan)} 列全为 NaN, 将其替换为 0")
            features[:, col_is_nan] = 0

        # 使用均值填充剩余的 NaN (只在非全 NaN 列上计算均值)
        if not np.all(col_is_nan): # 只有在存在非全 NaN 列时才计算均值
            col_mean = np.nanmean(features[:, ~col_is_nan], axis=0)
            # 找到 NaN 的位置，只在非全 NaN 列中查找
            inds_row, inds_col = np.where(np.isnan(features[:, ~col_is_nan]))
            # 使用对应列的均值进行填充
            features[inds_row, np.where(~col_is_nan)[0][inds_col]] = np.take(col_mean, inds_col)

        num_nan_after = np.isnan(features).sum()
        print(f"{name} - NaN数量（填充后）: {num_nan_after}")
        if num_nan_after > 0:
             print(f"{name} - 警告: 填充后仍存在 NaN 值。可能需要检查数据源或填充策略。")
    else:
        print(f"{name} - 无 NaN 值，无需填充。")

    return features


def visualize_embeddings(model, handcrafted_tensor, signal_tensor, y, prototypes, device, title_suffix="Feature Space", random_seed=42):
    """ Visualizes embeddings using t-SNE, handling potential perplexity issues. """
    model.eval()
    with torch.no_grad():
        # Ensure signal tensor has correct shape [N, 1, L]
        if signal_tensor.dim() == 2:
            signal_tensor = signal_tensor.unsqueeze(1)

        result = model(handcrafted_tensor.to(device), signal_tensor.to(device), domain_labels=None, lambda_grl=0.0)

        # Check model output structure
        if not isinstance(result, tuple) or len(result) < 3:
             print(f"Model output type: {type(result)}")
             if isinstance(result, tuple):
                 print(f"Model output length: {len(result)}")
             raise ValueError(f"Model output format unexpected. Expected tuple of length >= 3.")

        class_logits, domain_logits, fused_features = result[:3] # Take first 3 elements

        fused_features = fused_features.cpu().numpy()
        prototypes_np = prototypes.cpu().numpy()
        y_np = y # Assuming y is already a numpy array or tensor on CPU

        # Ensure y is numpy array
        if isinstance(y_np, torch.Tensor):
            y_np = y_np.cpu().numpy()


        print(f"Fused features shape: {fused_features.shape}")
        print(f"Prototypes shape: {prototypes_np.shape}")
        print(f"Labels shape: {y_np.shape}")

        # Check for empty inputs
        if fused_features.shape[0] == 0:
            print("警告: 没有融合特征可供可视化。")
            return
        if prototypes_np.shape[0] > 0 and fused_features.shape[1] != prototypes_np.shape[1]:
            print(f"警告：fused_features (dim {fused_features.shape[1]}) 和 prototypes_np (dim {prototypes_np.shape[1]}) 维度不匹配。无法添加原型到可视化。")
            all_features = fused_features
            all_labels = y_np
            plot_prototypes = False
        elif prototypes_np.shape[0] > 0:
             # Combine features and prototypes only if dimensions match
             all_features = np.vstack((fused_features, prototypes_np))
             # Create labels for prototypes (e.g., using unique class labels)
             prototype_labels = np.arange(prototypes_np.shape[0]) # Or use actual class labels if available
             all_labels = np.hstack((y_np, prototype_labels)) # Combine sample and prototype labels
             plot_prototypes = True
        else:
             all_features = fused_features
             all_labels = y_np
             plot_prototypes = False


        print(f"Total features for t-SNE: {all_features.shape}")

        # Robust imputation
        all_features = robust_imputation(all_features, name="all_features_for_tsne")

        # --- Fix for ValueError: perplexity must be less than n_samples ---
        n_samples = all_features.shape[0]
        if n_samples <= 1:
             print("警告: 样本数量过少 (<= 1)，无法进行 t-SNE 可视化。")
             return

        # Adjust perplexity based on n_samples
        perplexity_value = min(30.0, float(n_samples - 1))
        if perplexity_value < 5: # t-SNE works best with perplexity between 5 and 50
            print(f"警告: 样本数量 ({n_samples}) 非常小，使用的 perplexity ({perplexity_value:.1f}) 可能导致 t-SNE 效果不佳。")
        # Ensure perplexity is at least 1 if n_samples > 1
        perplexity_value = max(1.0, perplexity_value)


        print(f"Running t-SNE with n_samples={n_samples}, perplexity={perplexity_value:.1f}")
        tsne = TSNE(n_components=2, random_state=random_seed, perplexity=perplexity_value, n_iter=300, init='pca')
        try:
            embedded = tsne.fit_transform(all_features)
        except Exception as e:
            print(f"t-SNE 计算失败: {e}")
            # Fallback or alternative visualization could be added here if needed
            # Example: PCA
            # from sklearn.decomposition import PCA
            # print("t-SNE 失败，尝试使用 PCA 进行可视化...")
            # pca = PCA(n_components=2)
            # try:
            #     embedded = pca.fit_transform(all_features)
            # except Exception as pca_e:
            #     print(f"PCA 也失败了: {pca_e}")
            #     return # Give up if PCA also fails
            return # Give up if t-SNE fails


        # Plotting
        plt.figure(figsize=(10, 8))
        num_actual_samples = len(y_np)

        # Plot actual samples
        scatter_samples = plt.scatter(embedded[:num_actual_samples, 0], embedded[:num_actual_samples, 1], c=y_np, cmap='viridis', alpha=0.7, label='Samples')

        # Plot prototypes if they were included and dimensions matched
        if plot_prototypes:
             num_prototypes = prototypes_np.shape[0]
             # Ensure prototype labels match the color map if possible, otherwise use a distinct marker/color
             # Using unique colors might require adjusting the color map or handling manually
             plt.scatter(embedded[num_actual_samples:, 0], embedded[num_actual_samples:, 1], c='red', marker='X', s=100, label='Prototypes') # Simple red 'X' for now

        plt.title(f't-SNE Visualization of {title_suffix}')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        # Create a legend that handles both samples and prototypes
        handles, labels = plt.gca().get_legend_handles_labels()
        # Add legend for colormap if needed (more complex)
        # For now, just show 'Samples' and 'Prototypes'
        unique_labels = np.unique(y_np)
        if len(unique_labels) < 10: # Add individual class legend only if few classes
             legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=f'Class {int(l)}', markerfacecolor=scatter_samples.cmap(scatter_samples.norm(l)), markersize=8) for l in unique_labels]
             if plot_prototypes:
                 legend_elements.append(plt.Line2D([0], [0], marker='X', color='w', label='Prototypes', markerfacecolor='red', markersize=10))
             plt.legend(handles=legend_elements, title="Classes")
        else:
             plt.legend(handles=handles, title="Legend") # Default legend for many classes

        plt.show()


# (Keep other functions like visualize_cnn_features, analyze_feature_importance, etc.)
# Make sure to apply similar perplexity fix to visualize_cnn_features if you use it.

def visualize_cnn_features(model, signal_tensor, y, device, title_suffix="CNN Features", random_seed=42):
    """ Visualizes CNN features using t-SNE, handling potential perplexity issues. """
    model.eval()
    with torch.no_grad():
        # Ensure signal tensor has correct shape [N, 1, L]
        if signal_tensor.dim() == 2:
            signal_tensor = signal_tensor.unsqueeze(1)
        # Get CNN features - ensure model has 'signal_cnn' attribute
        if not hasattr(model, 'signal_cnn'):
             print("模型没有 'signal_cnn' 属性，无法可视化 CNN 特征。")
             return
        cnn_features = model.signal_cnn(signal_tensor.to(device))
        # Assume signal_cnn ends with Flatten, so output is [N, num_features]
        cnn_features = cnn_features.cpu().numpy()

        # Ensure y is numpy array
        if isinstance(y, torch.Tensor):
            y = y.cpu().numpy()

        # Check for empty inputs
        if cnn_features.shape[0] == 0:
            print("警告: 没有 CNN 特征可供可视化。")
            return

        print(f"CNN features shape: {cnn_features.shape}")
        print(f"Labels shape: {y.shape}")

        # Robust imputation
        cnn_features = robust_imputation(cnn_features, name="cnn_features_for_tsne")

        # --- Fix for ValueError: perplexity must be less than n_samples ---
        n_samples = cnn_features.shape[0]
        if n_samples <= 1:
             print("警告: 样本数量过少 (<= 1)，无法进行 t-SNE 可视化。")
             return

        # Adjust perplexity based on n_samples
        perplexity_value = min(30.0, float(n_samples - 1))
        if perplexity_value < 5:
            print(f"警告: 样本数量 ({n_samples}) 非常小，使用的 perplexity ({perplexity_value:.1f}) 可能导致 t-SNE 效果不佳。")
        perplexity_value = max(1.0, perplexity_value)

        print(f"Running t-SNE with n_samples={n_samples}, perplexity={perplexity_value:.1f}")
        tsne = TSNE(n_components=2, random_state=random_seed, perplexity=perplexity_value, n_iter=300, init='pca')
        try:
            embedded = tsne.fit_transform(cnn_features)
        except Exception as e:
            print(f"t-SNE 计算失败: {e}")
            return

        # Plotting
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(embedded[:, 0], embedded[:, 1], c=y, cmap='viridis', alpha=0.7)
        plt.title(f't-SNE Visualization of {title_suffix}')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')

        # Add colorbar or legend
        try: # Handle potential errors with colorbar/legend
            unique_labels = np.unique(y)
            if len(unique_labels) < 10: # Use legend for few classes
                 legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=f'Class {int(l)}', markerfacecolor=scatter.cmap(scatter.norm(l)), markersize=8) for l in unique_labels]
                 plt.legend(handles=legend_elements, title="Classes")
            else: # Use colorbar for many classes
                 plt.colorbar(scatter, label='Class Label')
        except Exception as legend_e:
            print(f"创建图例或颜色条时出错: {legend_e}")

        plt.show()

def plot_confusion_matrix(y_true, y_pred, classes):
    """
    绘制混淆矩阵
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(max(6, len(classes)//2), max(5, len(classes)//2.5)))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes, annot_kws={"size": 10})
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()


def calculate_metrics(y_true, y_pred):
    """
    计算评估指标 (F1-Score Weighted)
    """
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    print(f"F1-Score (Weighted): {f1:.4f}")
    return f1


def grl_lambda_schedule(epoch, total_epochs, alpha=10.0, verbose=False):
    """
    Calculates the GRL lambda dynamically based on training progress.
    Common DANN formula: lambda increases from 0 to 1 following sigmoid-like curve.
    """
    p = float(epoch) / total_epochs
    lambda_p = 2. / (1. + math.exp(-alpha * p)) - 1
    lambda_p = max(0.0, min(1.0, lambda_p)) # Clamp

    if verbose and (epoch % (total_epochs // 10) == 0 or epoch == total_epochs -1) :
        print(f"Epoch {epoch+1}/{total_epochs}: GRL Lambda = {lambda_p:.4f}")
    return lambda_p


# --- Updated plot_loss_curves function ---
def plot_loss_curves(train_losses, val_losses=None, class_losses=None, domain_losses=None, title='Loss Curves'):
    """
    绘制训练、验证、类别和领域损失曲线（如果提供）。

    Args:
        train_losses (list): List of total training losses per epoch/step.
        val_losses (list, optional): List of validation losses. Defaults to None.
        class_losses (list, optional): List of class losses. Defaults to None.
        domain_losses (list, optional): List of raw domain losses (before lambda weighting). Defaults to None.
        title (str, optional): Title for the plot. Defaults to 'Loss Curves'.
    """
    plt.figure(figsize=(12, 7)) # Wider figure to accommodate multiple lines

    # Plot total training loss
    plt.plot(train_losses, label='Total Train Loss', color='blue', linewidth=2)

    # Plot class loss if provided
    if class_losses:
        plt.plot(class_losses, label='Class Loss', color='green', linestyle='--')

    # Plot domain loss if provided
    # Filter out potential None values if domain loss wasn't calculated every epoch
    if domain_losses:
        # Find indices where domain loss was recorded (not None or NaN)
        valid_indices = [i for i, loss in enumerate(domain_losses) if loss is not None and not np.isnan(loss)]
        valid_losses = [domain_losses[i] for i in valid_indices]
        if valid_indices: # Only plot if there are valid domain losses
             plt.plot(valid_indices, valid_losses, label='Domain Loss (Raw)', color='red', linestyle=':')

    # Plot validation loss if provided
    if val_losses:
        plt.plot(val_losses, label='Validation Loss', color='orange', linewidth=2)


    plt.xlabel("Epoch / Logging Step")
    plt.ylabel("Loss Value")
    plt.legend()
    plt.title(title)
    plt.grid(True)
    plt.ylim(bottom=0) # Start y-axis at 0 for losses
    plt.tight_layout()
    plt.show()

# (Keep extract_handcrafted_from_signal if needed, though it seems unused in train/model)
# def extract_handcrafted_from_signal(filtered_signal):
#     # ... implementation ...