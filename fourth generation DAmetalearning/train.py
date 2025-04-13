# train.py
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from datapreprocessing import collect_fruit_data, prepare_dataset
from model import FusionDomainAdaptationNetwork
from utils import calculate_metrics, plot_confusion_matrix, visualize_embeddings

# -------------------------------
# 数据导入与构造数据集（使用手工特征，包含Fpeak）
# -------------------------------
data_paths = {
    "grape": {
        "ripe": r"D:/大二下学期/CDIO/ripemetalearning",
        "rotten": r"D:/大二下学期/CDIO/rottenmetalearning"
    },
    "strawberry": {
        "ripe": r"D:/大二下学期/CDIO/sberry_ripe",
        "rotten": r"D:/大二下学期/CDIO/sberry_rotten"
    },
}
domain_mapping = {"grape": 0, "strawberry": 1}
fruit_data_list = []
for fruit, paths in data_paths.items():
    handcrafted, labels, domain_labels, signals = collect_fruit_data(fruit, paths["ripe"], paths["rotten"],
                                                                     domain_mapping[fruit])
    fruit_data_list.append((handcrafted, labels, domain_labels, signals))
# 这里只需要使用手工特征
X_handcrafted, y_label, y_domain, _ = prepare_dataset(fruit_data_list)

# -------------------------------
# 数据集划分
# -------------------------------
X_hand_train, X_hand_test, y_label_train, y_label_test, y_domain_train, y_domain_test = train_test_split(
    X_handcrafted, y_label, y_domain, test_size=0.2, stratify=y_label, random_state=42)

# 转换为 Tensor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X_hand_train = torch.tensor(X_hand_train, dtype=torch.float32).to(device)
y_label_train = torch.tensor(y_label_train, dtype=torch.long).to(device)
y_domain_train = torch.tensor(y_domain_train, dtype=torch.long).to(device)

X_hand_test = torch.tensor(X_hand_test, dtype=torch.float32).to(device)
y_label_test = torch.tensor(y_label_test, dtype=torch.long).to(device)
y_domain_test = torch.tensor(y_domain_test, dtype=torch.long).to(device)

# -------------------------------
# 模型初始化
# -------------------------------
model = FusionDomainAdaptationNetwork(
    handcrafted_input_size=X_handcrafted.shape[1],
    handcrafted_hidden_size=64,
    handcrafted_feature_dim=32,
    fusion_hidden_size=64,
    dropout_rate=0.2,
    domain_classes=2,         # 水果种类： grape / strawberry
    domain_embed_dim=8
).to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
criterion_class = nn.CrossEntropyLoss()
criterion_domain = nn.CrossEntropyLoss()

# -------------------------------
# 训练模型
# -------------------------------
epochs = 100
lambda_domain = 0.4

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    # 仅使用手工特征作为输入
    class_logits, domain_logits, features = model(X_hand_train, domain_labels=y_domain_train, lambda_grl=1.0)
    loss_class = criterion_class(class_logits, y_label_train)
    loss_domain = criterion_domain(domain_logits, y_domain_train)
    loss = loss_class + lambda_domain * loss_domain
    loss.backward()
    optimizer.step()
    print(f"[Epoch {epoch+1}/{epochs}] Loss: {loss.item():.4f}  Class Loss: {loss_class.item():.4f}  Domain Loss: {loss_domain.item():.4f}")

# -------------------------------
# 测试与评估
# -------------------------------
model.eval()
with torch.no_grad():
    class_logits, domain_logits, features = model(X_hand_test, domain_labels=y_domain_test, lambda_grl=0.0)
    y_pred_label = torch.argmax(class_logits, dim=1).cpu().numpy()
    y_pred_domain = torch.argmax(domain_logits, dim=1).cpu().numpy()

y_true_label = y_label_test.cpu().numpy()
y_true_domain = y_domain_test.cpu().numpy()

print("\n🍇 水果种类识别结果（Domain分类）:")
print(classification_report(y_true_domain, y_pred_domain, target_names=["Grape", "Strawberry"]))
print("准确率:", accuracy_score(y_true_domain, y_pred_domain))
print("F1-score:", f1_score(y_true_domain, y_pred_domain, average="weighted"))

print("\n🍓 成熟度识别结果（Label分类）:")
print(classification_report(y_true_label, y_pred_label, target_names=["Ripe", "Rotten"]))
print("准确率:", accuracy_score(y_true_label, y_pred_label))
print("F1-score:", f1_score(y_true_label, y_pred_label, average="weighted"))

# 如果需要可视化融合特征，可以构造原型后调用 visualize_embeddings
prototypes = torch.zeros(2, features.shape[1]).to(device)  # 简单示例：2 个类别，每个原型置 0（实际需用训练样本计算原型）
# visualize_embeddings(model, X_hand_test, None, y_true_label, prototypes, device)
