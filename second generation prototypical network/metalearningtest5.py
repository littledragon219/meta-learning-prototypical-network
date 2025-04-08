# 较test4，这个多了ROC等曲线输出
import os
import pandas as pd
import numpy as np
from scipy.fft import fft, fftfreq
import torch
import torch.nn as nn
import pickle
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


def extract_force_data_and_calculate_stats(folder_path):
    all_stats, all_fft_data, all_labels = [], [], []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)
                try:
                    df = pd.read_csv(file_path)
                    force_data = df.iloc[:, 2].values
                    force_data = force_data[force_data >= 0]

                    if len(force_data) == 0:
                        continue

                    mean_force = np.mean(force_data)
                    std_force = np.std(force_data)
                    rising_rate = 0
                    if len(force_data) > 1:
                        diff_force = np.diff(force_data)
                        if len(diff_force) > 0:
                            rising_rate = np.mean(diff_force[diff_force > 0]) if np.sum(diff_force > 0) > 0 else 0
                    peak_force = np.max(force_data)

                    N = len(force_data)
                    T = 0.1
                    yf = fft(force_data)
                    xf = fftfreq(N, T)[:N // 2]
                    yf_abs = 2.0 / N * np.abs(yf[:N // 2])

                    stats = {
                        'file': file,
                        'mean_force': mean_force,
                        'std_force': std_force,
                        'rising_rate': rising_rate,
                        'peak_force': peak_force
                    }
                    all_stats.append(stats)
                    all_fft_data.append(yf_abs)

                    if 'ripe' in root.lower():
                        label = 0
                    elif 'rotten' in root.lower():
                        label = 1
                    else:
                        continue
                    all_labels.append(label)
                except (pd.errors.ParserError, Exception):
                    continue
    return all_stats, all_fft_data, all_labels


def prepare_dataset(stats_folder, fft_data_folder, fixed_fft_length):
    X = []
    for stats, fft_data in zip(stats_folder, fft_data_folder):
        features = [stats['mean_force'], stats['std_force'],
                    stats['rising_rate'], stats['peak_force']]
        if len(fft_data) > fixed_fft_length:
            padded_fft_data = fft_data[:fixed_fft_length]
        else:
            padded_fft_data = np.pad(
                fft_data, (0, fixed_fft_length - len(fft_data)), mode='constant')
        features.extend(padded_fft_data)
        X.append(features)
    return np.array(X)  


class PrototypicalNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0.5):
        super(PrototypicalNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def main():
    model_filename = 'best_prototypical_model.pkl'
    max_fft_length_filename = 'metamax_fft_length.pkl'
    prototypes_filename = 'final_prototypes.pkl'

    try:
        with open(model_filename, 'rb') as f:
            model_state_dict = pickle.load(f)
        with open(max_fft_length_filename, 'rb') as f:
            model_max_fft_length = pickle.load(f)
        with open(prototypes_filename, 'rb') as f:
            prototypes = torch.tensor(pickle.load(f), dtype=torch.float32)
    except FileNotFoundError:
        return
    except Exception:
        return

    input_size = 4 + model_max_fft_length
    hidden_size = 32
    output_size = 16
    dropout_rate = 0.5
    model = PrototypicalNetwork(input_size, hidden_size, output_size, dropout_rate)
    model.load_state_dict(model_state_dict)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    prototypes = prototypes.to(device)

    new_folder = r'd:\大二下学期\cdio\test'

    new_stats_folder, new_fft_data_folder, y_true = extract_force_data_and_calculate_stats(new_folder)

    X_new = prepare_dataset(
        new_stats_folder, new_fft_data_folder,
        fixed_fft_length=model_max_fft_length
    )

    X_new_tensor = torch.tensor(X_new, dtype=torch.float32).to(device)

    model.eval()
    with torch.no_grad():
        query_features = model(X_new_tensor)
        distances = torch.cdist(query_features, prototypes)
        y_pred = torch.argmin(distances, dim=1).cpu().numpy()

    accuracy = np.mean(y_pred == y_true)
    print(f"模型在测试集上的准确率: {accuracy:.4f}")

    # 生成混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Ripe', 'Rotten'])
    plt.yticks(tick_marks, ['Ripe', 'Rotten'])

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

    # 生成ROC曲线
    if len(np.unique(y_true)) == 2:
        distances_to_rotten = distances[:, 1].cpu().numpy()
        fpr, tpr, thresholds = roc_curve(y_true, -distances_to_rotten)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('metalearning Receiver operating characteristic')
        plt.legend(loc="lower right")
        plt.show()


if __name__ == "__main__":
    main()