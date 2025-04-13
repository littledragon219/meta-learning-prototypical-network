# train_fusion.py
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler
import joblib

# Import necessary components from project files
from datapreprocessing import collect_fruit_data, EXPECTED_LENGTH # Ensure EXPECTED_LENGTH is correct
from model import FusionDomainAdaptationNetwork # Import the updated model
from utils import calculate_metrics, plot_confusion_matrix, visualize_embeddings, plot_loss_curves, grl_lambda_schedule # Import schedule

np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# --- Configuration ---
data_paths = {
    "grape": {
        "ripe": r"D:/大二下学期/CDIO/grape_ripemetalearning",
        "rotten": r"D:/大二下学期/CDIO/grape_rottenmetalearning"
    },
    "strawberry": {
        "ripe": r"D:/大二下学期/CDIO/strawberry_ripe",
        "rotten": r"D:/大二下学期/CDIO/strawberry_rotten"
    },
    "tomato": {
        "ripe": r"D:/大二下学期/CDIO/ctomato/tomato_ripe", # CHANGE THIS PATH
        "rotten": r"D:/大二下学期/CDIO/ctomato/tomato_rotten"
    },
}
MODEL_SAVE_PATH = 'final_model_and_config_v2.pth'
SCALER_SAVE_PATH = 'scaler_v2.joblib'
EPOCHS = 200
LEARNING_RATE = 0.005
WEIGHT_DECAY = 1e-5
GRL_ALPHA = 0.8

# --- Model Hyperparameters ---
HANDCRAFTED_HIDDEN_SIZE = 128
CNN_CHANNELS = [16, 32]
CNN_KERNELS = [5, 3]
CNN_POOLS = [2, 2]
CNN_ATTENTION_DIM = 128
FUSION_INTERACTION_DIM = 64
FUSION_HIDDEN_SIZE = 128
DROPOUT_RATE = 0.5
DOMAIN_EMBED_DIM = 8
DOMAIN_HIDDEN_DIM = 64
# --- Data Loading ---
print("开始加载和预处理数据...")
fruit_names = list(data_paths.keys())
domain_mapping = {name: i for i, name in enumerate(fruit_names)}
domain_classes_count = len(fruit_names)
fruit_data_list = []
num_maturity_classes = 0
signals_list_for_length_check = []

for fruit, paths in data_paths.items():
    print(f"加载水果: {fruit} (领域 ID: {domain_mapping[fruit]})")
    try:
        handcrafted, labels, domain_labels, signals = collect_fruit_data(fruit, paths["ripe"], paths["rotten"], domain_mapping[fruit])
        if handcrafted.shape[0] > 0:
            fruit_data_list.append((handcrafted, labels, domain_labels, signals))
            signals_list_for_length_check.append(signals)
            if len(labels) > 0:
                 current_max_label = np.max(labels)
                 num_maturity_classes = max(num_maturity_classes, current_max_label + 1)
            print(f"  - 加载了 {handcrafted.shape[0]} 个样本.")
        else:
            print(f"  - 警告: 未能从 {fruit} 的路径加载任何数据。")
    except Exception as e:
        print(f"  - 错误: 加载 {fruit} 数据时出错: {e}")
        continue

if not fruit_data_list:
     raise ValueError("未能从任何指定路径加载数据。请检查 data_paths 和数据文件。")

print(f"总共加载了 {len(fruit_data_list)} 种水果的数据。")
print(f"检测到的成熟度类别数量: {num_maturity_classes}")
print(f"检测到的领域类别数量: {domain_classes_count}")

try:
    cnn_input_len = EXPECTED_LENGTH # Get expected length
    print(f"使用的信号长度 (用于CNN): {cnn_input_len}")
except NameError:
    print("错误: 无法从 datapreprocessing 确定 EXPECTED_LENGTH。请确保它已定义并导入。")
    if signals_list_for_length_check:
         cnn_input_len = signals_list_for_length_check[0].shape[1]
         print(f"警告: 使用第一个加载信号的长度作为 cnn_input_len: {cnn_input_len}")
    else:
         raise ValueError("无法确定 CNN 输入长度。")

# --- Combine, Scale, Save Scaler ---
print("合并数据集并拟合 Scaler...")
all_handcrafted = []
all_labels = []
all_domain_labels = []
all_signals = []
for handcrafted, labels, domain_labels, signals in fruit_data_list:
    if signals.shape[1] != cnn_input_len:
        print(f"  - 警告: 信号长度 {signals.shape[1]} 与期望长度 {cnn_input_len} 不符。将进行调整。")
        if signals.shape[1] > cnn_input_len:
            signals = signals[:, :cnn_input_len]
        else:
            signals = np.pad(signals, ((0, 0), (0, cnn_input_len - signals.shape[1])), 'constant', constant_values=0)
    all_handcrafted.append(handcrafted)
    all_labels.append(labels)
    all_domain_labels.append(domain_labels)
    all_signals.append(signals)
X_handcrafted_combined = np.vstack(all_handcrafted)
y_label_combined = np.hstack(all_labels)
y_domain_combined = np.hstack(all_domain_labels)
signals_combined = np.vstack(all_signals)

scaler = StandardScaler()
X_handcrafted_scaled = scaler.fit_transform(X_handcrafted_combined)
print(f"保存 Scaler 到: {SCALER_SAVE_PATH}")
joblib.dump(scaler, SCALER_SAVE_PATH)
handcrafted_input_dim = X_handcrafted_scaled.shape[1]
print(f"最终手工特征维度 (包括 Fpeak): {handcrafted_input_dim}")

# --- Data Split and Tensor Conversion ---
print("划分训练集和测试集...")
X_hand_train, X_hand_test, y_label_train, y_label_test, y_domain_train, y_domain_test, signals_train, signals_test = train_test_split(
    X_handcrafted_scaled, y_label_combined, y_domain_combined, signals_combined,
    test_size=0.2, stratify=y_label_combined, random_state=42
)
print(f"训练集大小: {X_hand_train.shape[0]}, 测试集大小: {X_hand_test.shape[0]}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")
X_hand_train_t = torch.tensor(X_hand_train, dtype=torch.float32).to(device)
signals_train_t = torch.tensor(signals_train, dtype=torch.float32).unsqueeze(1).to(device)
y_label_train_t = torch.tensor(y_label_train, dtype=torch.long).to(device)
y_domain_train_t = torch.tensor(y_domain_train, dtype=torch.long).to(device)
X_hand_test_t = torch.tensor(X_hand_test, dtype=torch.float32).to(device)
signals_test_t = torch.tensor(signals_test, dtype=torch.float32).unsqueeze(1).to(device)
y_label_test_t = torch.tensor(y_label_test, dtype=torch.long).to(device)
y_domain_test_t = torch.tensor(y_domain_test, dtype=torch.long).to(device)

# --- Model Initialization (Add 'cnn_input_length' to config) ---
print("初始化新模型...")
model_config = {
    'handcrafted_input_size': handcrafted_input_dim,
    'handcrafted_hidden_size': HANDCRAFTED_HIDDEN_SIZE,
    # --- CNN Params ---
    'cnn_channels': CNN_CHANNELS,
    'cnn_kernels': CNN_KERNELS,
    'cnn_pools': CNN_POOLS,
    # --- Added Key ---
    'cnn_input_length': cnn_input_len, # **** ADDED THIS LINE ****
    # --- Attention Param ---
    'cnn_attention_dim': CNN_ATTENTION_DIM,
    # --- Fusion Params ---
    'fusion_interaction_dim': FUSION_INTERACTION_DIM,
    'fusion_hidden_size': FUSION_HIDDEN_SIZE,
    'dropout_rate': DROPOUT_RATE,
    # --- Output/Domain Params ---
    'num_classes': num_maturity_classes,
    'domain_classes': domain_classes_count,
    'domain_embed_dim': DOMAIN_EMBED_DIM,
    'domain_hidden_dim': DOMAIN_HIDDEN_DIM
}

model = FusionDomainAdaptationNetwork(**model_config).to(device)
print(model)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"模型总可训练参数: {total_params:,}")

# --- Optimizer and Loss ---
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
criterion_class = nn.CrossEntropyLoss()
criterion_domain = nn.CrossEntropyLoss()

# --- Training Loop (Store class_loss and use updated plotting) ---
print(f"\n开始训练 {EPOCHS} 个 Epochs...")
# Initialize lists to store losses per epoch
train_losses = []   # Stores total loss
class_losses = []   # Stores class loss
domain_losses = []  # Stores raw domain loss (before lambda weighting)

for epoch in range(EPOCHS):
    model.train()
    optimizer.zero_grad()
    current_lambda = grl_lambda_schedule(epoch, EPOCHS, alpha=GRL_ALPHA, verbose=(epoch==0))
    class_logits, domain_logits, _ = model(X_hand_train_t, signals_train_t, domain_labels=y_domain_train_t, lambda_grl=current_lambda)

    # Calculate losses
    loss_class = criterion_class(class_logits, y_label_train_t)
    loss = loss_class # Start with class loss
    loss_domain_val = None # Use None if domain loss isn't calculated

    if domain_logits is not None:
        loss_domain = criterion_domain(domain_logits, y_domain_train_t)
        loss = loss + current_lambda * loss_domain # Add weighted domain loss to total loss
        loss_domain_val = loss_domain.item() # Get raw domain loss value

    loss.backward()
    optimizer.step()

    # Store losses for plotting
    train_losses.append(loss.item())
    class_losses.append(loss_class.item())
    domain_losses.append(loss_domain_val) # Appends the value or None

    # Logging (Keep as before)
    if (epoch + 1) % 10 == 0 or epoch == EPOCHS - 1:
        log_msg = f"[Epoch {epoch+1}/{EPOCHS}] Train Loss: {loss.item():.4f} (Class: {loss_class.item():.4f}"
        # Display raw domain loss value in log if available
        if loss_domain_val is not None: log_msg += f", Domain: {loss_domain_val:.4f}, Lambda: {current_lambda:.3f})"
        else: log_msg += ")"
        print(log_msg)
# --- Plot Loss Curves (Use updated function call) ---
print("\n绘制损失曲线...")
plot_loss_curves(
    train_losses=train_losses,
    class_losses=class_losses,
    domain_losses=domain_losses, # Pass the raw domain losses
    title='Training Loss Curves (Total, Class, Domain)'
)

# --- Save Final Model & Config (Keep as before) ---
print(f"\n训练完成。保存最终模型状态和配置到: {MODEL_SAVE_PATH}")
save_data = {'config': model_config, 'state_dict': model.state_dict()}
torch.save(save_data, MODEL_SAVE_PATH)
print("模型和配置已保存。")

# --- Final Evaluation (Keep as before) ---
print("\n在测试集上评估最终模型...")
model.eval()
with torch.no_grad():
    class_logits_test, domain_logits_test, fused_feats_test = model(X_hand_test_t, signals_test_t, domain_labels=y_domain_test_t, lambda_grl=0.0)
    y_pred_label = torch.argmax(class_logits_test, dim=1).cpu().numpy()
    y_pred_domain = None
    if domain_logits_test is not None: y_pred_domain = torch.argmax(domain_logits_test, dim=1).cpu().numpy()
y_true_label_np = y_label_test_t.cpu().numpy(); y_true_domain_np = y_domain_test_t.cpu().numpy()
label_names = []; maturity_states = ["Ripe", "Rotten"]; num_maturity_states = len(maturity_states)
for i in range(num_maturity_classes):
    fruit_index = i // num_maturity_states; maturity_index = i % num_maturity_states
    if fruit_index < len(fruit_names): label_names.append(f"{maturity_states[maturity_index]} {fruit_names[fruit_index].capitalize()}")
    else: label_names.append(f"Class {i}")
domain_names = [name.capitalize() for name in fruit_names]
print("\n--- 最终模型评估结果 (测试集) ---")
if y_pred_domain is not None:
    print(f"\n🍇 领域分类 ({'/'.join(domain_names)}):")
    print(classification_report(y_true_domain_np, y_pred_domain, target_names=domain_names, zero_division=0))
    print(f"领域分类准确率: {accuracy_score(y_true_domain_np, y_pred_domain):.4f}")
    plot_confusion_matrix(y_true_domain_np, y_pred_domain, classes=domain_names)
print(f"\n🍓 类别分类 ({'/'.join(label_names[:4])}...):")
print(classification_report(y_true_label_np, y_pred_label, target_names=label_names, zero_division=0))
print(f"类别分类准确率: {accuracy_score(y_true_label_np, y_pred_label):.4f}")
calculate_metrics(y_true_label_np, y_pred_label)
plot_confusion_matrix(y_true_label_np, y_pred_label, classes=label_names)

# --- Visualization (Keep as before) ---
print("\n进行最终模型在测试集上的 t-SNE 可视化...")
if fused_feats_test is not None and fused_feats_test.shape[0] > 0:
    prototypes_vis = torch.zeros(num_maturity_classes, fused_feats_test.shape[1]).to(device)
    visualize_embeddings(
        model=model, handcrafted_tensor=X_hand_test_t, signal_tensor=signals_test_t,
        y=y_true_label_np, prototypes=prototypes_vis, device=device,
        title_suffix="Test Set Embeddings (Final Model V2)"
    )
else: print("无法进行 t-SNE 可视化，测试集融合特征为空或不存在。")

print("\n训练和评估流程结束。")