import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import torch
from sklearn.metrics import f1_score, confusion_matrix
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

def robust_imputation(features, name="features"):
    """
    对输入的特征矩阵进行缺失值填充：
      - 对于某列全为 NaN，则替换为0，
      - 其余部分采用 SimpleImputer 处理。
    """
    num_nan_before = np.isnan(features).sum()
    print(f"{name} - NaN数量（填充前）: {num_nan_before}")
    col_mean = np.nanmean(features, axis=0)
    cols_nan = np.isnan(col_mean)
    if np.any(cols_nan):
        print(f"{name} - 检测到全为 NaN 的列, 将其替换为0")
        col_mean[cols_nan] = 0
    inds = np.where(np.isnan(features))
    features[inds] = np.take(col_mean, inds[1])
    num_nan_after = np.isnan(features).sum()
    print(f"{name} - NaN数量（填充后）: {num_nan_after}")
    return features

def visualize_embeddings(model, handcrafted_tensor, signal_tensor, y, prototypes, device, random_seed=42):
    """
    使用 t-SNE 可视化模型融合后的手工特征。
    注意：如果模型只用手工特征，此处 signal_tensor 可以传 None 或任意占位（不会被用到）。
    """
    model.eval()
    with torch.no_grad():
        # 模型返回值为 (class_logits, domain_logits, fused_features)
        result = model(handcrafted_tensor.to(device), signal_tensor.to(device) if signal_tensor is not None else None, domain_labels=None, lambda_grl=0.0)
        if len(result) == 3:
            _, _, fused_features = result
        else:
            raise ValueError(f"model 返回值数量为 {len(result)}，期望为 3")
        fused_features = fused_features.cpu().numpy()
        prototypes_np = prototypes.cpu().numpy()
        # 确保特征维度一致
        if fused_features.shape[1] != prototypes_np.shape[1]:
            print(f"警告：特征维度不一致，尝试调整原型维度。")
            if prototypes_np.shape[1] < fused_features.shape[1]:
                new_prototypes = np.zeros((prototypes_np.shape[0], fused_features.shape[1]))
                new_prototypes[:, :prototypes_np.shape[1]] = prototypes_np
                prototypes_np = new_prototypes
            else:
                print("无法调整原型维度，请检查计算。")
                return
        all_features = np.vstack((fused_features, prototypes_np))
        all_labels = np.hstack((y, np.unique(y)))
        all_features = robust_imputation(all_features, name="all_features")
        embedded = TSNE(n_components=2, random_state=random_seed).fit_transform(all_features)
        plt.scatter(embedded[:len(y), 0], embedded[:len(y), 1], c=y, cmap='viridis', label='Samples')
        plt.scatter(embedded[len(y):, 0], embedded[len(y):, 1], c='r', marker='X', label='Prototypes')
        plt.legend()
        plt.title('t-SNE Visualization of Feature Space')
        plt.show()

def analyze_feature_importance(X, y, feature_names=None):
    """
    利用随机森林分析特征重要性并绘图。
    """
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    importances = rf.feature_importances_
    if feature_names is None:
        feature_names = [f'Feature {i}' for i in range(X.shape[1])]
    plt.figure(figsize=(10, 6))
    indices = np.argsort(importances)[::-1]
    plt.title('Feature Importances')
    plt.bar(range(X.shape[1]), importances[indices], align='center')
    plt.xticks(range(X.shape[1]), [feature_names[i] for i in indices], rotation=45)
    plt.tight_layout()
    plt.show()
    return importances

def plot_confusion_matrix(y_true, y_pred, classes):
    """
    绘制混淆矩阵。
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()

def calculate_metrics(y_true, y_pred):
    """
    计算 F1-score。
    """
    f1 = f1_score(y_true, y_pred, average='weighted')
    print(f"F1-Score: {f1:.4f}")
    return f1

def plot_loss_curves(train_losses, val_losses):
    """
    绘制训练和验证损失曲线。
    """
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title('Training and Validation Loss Curves')
    plt.show()

def extract_handcrafted_from_signal(filtered_signal):
    """
    提取原始手工特征（此处不包含 Fpeak，因为在新的特征提取中已增加Fpeak特征）。
    如果只用手工特征，你可以保留其他部分作为参考。
    """
    initial_slope = filtered_signal[1] - filtered_signal[0]
    slope_change_rate = np.mean(np.diff(filtered_signal))
    variance = np.var(filtered_signal)
    norm_energy = np.sum(filtered_signal ** 2) / len(filtered_signal)
    return [initial_slope, slope_change_rate, variance, norm_energy]

