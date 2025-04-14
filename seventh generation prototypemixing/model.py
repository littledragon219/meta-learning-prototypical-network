# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureExtractor1DCNN(nn.Module):
    """Extracts features from raw 1D signal using CNN."""
    def __init__(self, input_channels=1, sequence_length=525, embedding_dim=64): # Default sequence length adjusted
        super().__init__()

        # --- OPTIMIZATION FOR CPU ---
        # Option 1: Keep original layers, but potentially reduce channels/embedding_dim in config.py
        # Option 2: Simplify the architecture (fewer layers/channels) below.
        # Let's try a slightly simplified version:
        # channels = [16, 32] # Reduced from [16, 32, 64]
        # kernels = [7, 5] # Reduced from [7, 5, 3]

        channels = [32, 64, 128] # Original channels
        kernels = [7, 5, 3] # Original kernels

        self.conv_blocks = nn.ModuleList()
        current_channels = input_channels
        current_seq_len = sequence_length

        for i, (ch, ks) in enumerate(zip(channels, kernels)):
            stride = 2 if i < len(channels) -1 else 1 # Stride 2 usually for downsampling blocks
            padding = ks // 2
            conv = nn.Conv1d(current_channels, ch, kernel_size=ks, stride=stride, padding=padding)
            bn = nn.BatchNorm1d(ch)
            # Pool on all but the last conv block
            pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1) if i < len(channels) - 1 else nn.Identity()

            self.conv_blocks.append(nn.Sequential(conv, bn, nn.ReLU(), pool))
            current_channels = ch
            # Calculate output length (important!)
            current_seq_len = (current_seq_len + 2 * padding - ks) // stride + 1 # Conv
            if i < len(channels) - 1: # Pool
                 current_seq_len = (current_seq_len + 2 * 1 - 3) // 2 + 1 # Pool


        flattened_size = int(current_seq_len * current_channels) # Calculate final flattened size

        self.fc = nn.Linear(flattened_size, embedding_dim)
        self.dropout = nn.Dropout(0.35) # Keep dropout moderate

        print(f"CNN Initialized: Input Seq Len={sequence_length}, Output Embedding Dim={embedding_dim}")
        print(f"CNN Architecture: Channels={channels}, Kernels={kernels}")
        print(f"CNN Flattened Size before FC: {flattened_size}")


    def forward(self, x):
        # x shape: (batch_size, channels, sequence_length)
        if x.ndim == 2: # Add channel dimension if missing (B, L) -> (B, 1, L)
             x = x.unsqueeze(1)
        # Input check:
        # if x.shape[2] != self.sequence_length: # Be careful if sequence length varies
        #     print(f"Warning: Input sequence length {x.shape[2]} differs from expected {self.sequence_length}")
             # Handle mismatch if necessary (e.g., pooling, padding - but ideally data prep fixes this)

        for block in self.conv_blocks:
            x = block(x)
            # print(f"Shape after block: {x.shape}") # Debugging shape changes


        # Check shape before flattening
        # print(f"Shape before flatten: {x.shape}")
        try:
            x = torch.flatten(x, 1) # Flatten all dimensions except batch
        except RuntimeError as e:
            print(f"Error during flatten: {e}. Input shape={x.shape}")
            # Maybe return zeros or raise error?
            raise e # Re-raise to stop execution

        # print(f"Shape after flatten: {x.shape}")

        x = self.dropout(x)
        x = self.fc(x)
        return x

class HandcraftedFeatureExtractor(nn.Module):
    """Extracts features from pre-computed handcrafted features using MLP."""
    def __init__(self, input_dim=5, hidden_dim=32, embedding_dim=64):
        super().__init__()
        # OPTIMIZATION: Simpler MLP might be sufficient
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3) # Moderate dropout
        self.fc2 = nn.Linear(hidden_dim, embedding_dim)
        print(f"Handcrafted MLP Initialized: Input Dim={input_dim}, Hidden={hidden_dim}, Embedding Dim={embedding_dim}")


    def forward(self, x):
        # x shape: (batch_size, num_handcrafted_features)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# --- Stage 1 Model ---
# Modified: No separate ProtoNet model needed, just use the extractor directly
# The training scripts will handle prototype calculation and loss.

# --- Stage 2 Model ---
class MaturityClassifier(nn.Module):
    """Classifies maturity based on input features (e.g., from Stage 1 extractor)."""
    def __init__(self, input_dim, hidden_dim=32, num_classes=2, dropout_rate=0.4):
        super().__init__()
        # OPTIMIZATION: Keep this relatively simple for speed
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate) # Slightly higher dropout can help
        # Optional: Add more layers if needed, but increases cost
        # self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        # self.relu2 = nn.ReLU()
        # self.dropout2 = nn.Dropout(dropout_rate)
        self.fc_out = nn.Linear(hidden_dim, num_classes)
        print(f"Maturity Classifier Initialized: Input Dim={input_dim}, Hidden={hidden_dim}, Classes={num_classes}")


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