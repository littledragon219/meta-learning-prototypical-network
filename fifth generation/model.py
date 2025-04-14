# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureExtractor1DCNN(nn.Module):
    """Extracts features from raw 1D signal using CNN."""
    def __init__(self, input_channels=1, sequence_length=1050, embedding_dim=64):
        super().__init__()
        # Simplified CNN - Adjust layers, channels, kernel sizes based on experimentation
        self.conv1 = nn.Conv1d(input_channels, 16, kernel_size=7, stride=2, padding=3) # Output len: seq_len / 2
        self.bn1 = nn.BatchNorm1d(16)
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1) # Output len: len / 2 = seq_len / 4

        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2) # Output len: len / 2 = seq_len / 8
        self.bn2 = nn.BatchNorm1d(32)
        self.pool2 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1) # Output len: len / 2 = seq_len / 16

        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1) # Output len: seq_len / 16
        self.bn3 = nn.BatchNorm1d(64)
        # Calculate the flattened size after conv layers
        # Example: sequence_length = 1050 -> 1050/16 = 65.625 -> floor = 65
        flattened_size = self._get_conv_output_size(sequence_length)

        self.fc = nn.Linear(flattened_size * 64, embedding_dim)
        self.dropout = nn.Dropout(0.3)

    def _get_conv_output_size(self, seq_len):
        # Helper to calculate size dynamically (or pre-calculate)
        len_out = seq_len
        # Conv1 + Pool1
        len_out = (len_out + 2*3 - 7) // 2 + 1
        len_out = (len_out + 2*1 - 3) // 2 + 1
        # Conv2 + Pool2
        len_out = (len_out + 2*2 - 5) // 2 + 1
        len_out = (len_out + 2*1 - 3) // 2 + 1
        # Conv3 (no pooling)
        len_out = (len_out + 2*1 - 3) // 1 + 1
        return int(len_out) # Ensure integer

    def forward(self, x):
        # x shape: (batch_size, channels, sequence_length)
        if x.ndim == 2: # Add channel dimension if missing
             x = x.unsqueeze(1)

        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))

        x = torch.flatten(x, 1) # Flatten all dimensions except batch
        x = self.dropout(x)
        x = self.fc(x)
        return x

class HandcraftedFeatureExtractor(nn.Module):
    """Extracts features from pre-computed handcrafted features using MLP."""
    def __init__(self, input_dim=5, hidden_dim=32, embedding_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_dim, embedding_dim)

    def forward(self, x):
        # x shape: (batch_size, num_handcrafted_features)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# --- Stage 1 Model ---
class ProtoNetDomainClassifier(nn.Module):
    """A wrapper for the feature extractor used in Prototypical Networks.
       The forward pass simply returns the embeddings.
       Prototype calculation and loss happen outside this module.
    """
    def __init__(self, feature_extractor):
        super().__init__()
        self.feature_extractor = feature_extractor

    def forward(self, x):
        # x can be raw signal (for CNN) or handcrafted features (for MLP)
        return self.feature_extractor(x)

# --- Stage 2 Model ---
class MaturityClassifier(nn.Module):
    """Classifies maturity based on input features (e.g., from Stage 1 extractor)."""
    def __init__(self, input_dim, hidden_dim=32, num_classes=2, dropout_rate=0.4):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        # Optional: Add more layers if needed
        # self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        # self.relu2 = nn.ReLU()
        # self.dropout2 = nn.Dropout(dropout_rate)
        self.fc_out = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        # if hasattr(self, 'fc2'):
        #     x = self.fc2(x)
        #     x = self.relu2(x)
        #     x = self.dropout2(x)
        logits = self.fc_out(x)
        return logits # Return logits directly