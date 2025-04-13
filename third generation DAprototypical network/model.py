# model.py
'''SignalCNN 模块
为了突出局部冲击变化，本版本的 CNN 部分仅使用了 2 个残差块和两次池化，重点捕捉信号冲击时段的局部特征。

TemporalAttention 模块
利用领域嵌入和全连接层计算基本注意力，然后根据 CNN 特征的 L2 范数与设定阈值计算“冲击因子”，使得在冲击区域注意力更高。

动态 GRL
函数 grl_lambda_schedule 可以在训练循环中根据当前迭代步数动态计算 lambda 值，从而使领域对抗逐步加强。

交互层
将手工特征和自动提取的 CNN 特征分别通过全连接映射到同一维度，然后做逐元素乘积，再与原始两部分特征拼接，为后续融合提供了更丰富的交互信息。
关注局部冲击特征
在 TemporalAttention 模块中，加入了一个基于 CNN 特征“幅值”计算的“冲击因子”（shock factor）。当局部信号较强时，注意力加权提高，从而使网络在冲击时段能够获得更多关注；而在稳定区域，则注意力衰减。

GRL 的动态调整
我们增加了一个 GRL lambda 动态调度函数（grl_lambda_schedule），它根据当前迭代步数动态计算 lambda。训练循环中可调用该函数更新 lambda 值。

交互层增强融合
在融合模块中，我们增加了一个交互层，分别将手工特征和 CNN 提取的特征映射到同一交互空间，再做逐元素乘积，并与原始两部分特征拼接。这种交互层有助于捕捉两类特征之间的相互依赖关系。
'''
# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import math

# --- Gradient Reversal Layer (Unchanged) ---
class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.lambda_
        return output, None

class GradientReversalLayer(nn.Module):
    def __init__(self): # Lambda is now passed in forward
        super().__init__()

    def forward(self, x, lambda_):
        return GradientReversalFunction.apply(x, lambda_)

# --- Residual Block for CNN ---
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample # To match dimensions if stride or channels change

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity # Skip connection
        out = self.relu(out)
        return out

# --- Temporal Attention Module ---
class TemporalAttention(nn.Module):
    def __init__(self, feature_dim, attention_dim):
        super(TemporalAttention, self).__init__()
        self.attention_fc = nn.Linear(feature_dim, attention_dim)
        self.value_fc = nn.Linear(feature_dim, feature_dim) # To compute values
        self.query_vec = nn.Parameter(torch.randn(1, attention_dim), requires_grad=True) # Learnable query vector

    def forward(self, x):
        # x shape: [N, C, L] (Batch, Channels/Features, Length)
        N, C, L = x.shape

        # Permute to [N, L, C] for linear layers
        x_permuted = x.permute(0, 2, 1)

        # Calculate attention scores based on learnable query
        # Project features to attention dim
        proj_key = torch.tanh(self.attention_fc(x_permuted)) # [N, L, attention_dim]

        # Calculate alignment scores (dot product with query vector)
        # query_vec: [1, attention_dim] -> expand to [N, 1, attention_dim]
        query_expanded = self.query_vec.expand(N, -1, -1) # [N, 1, attention_dim]
        # Scores: [N, 1, attention_dim] @ [N, attention_dim, L] -> [N, 1, L]
        scores = torch.bmm(query_expanded, proj_key.transpose(1, 2)) # [N, 1, L]

        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=2) # [N, 1, L]

        # Calculate values (can be identity or a projection)
        values = self.value_fc(x_permuted) # [N, L, C] (projected values)
        # Alternatively: values = x_permuted # Use original features as values

        # Apply attention weights to values
        # weights: [N, 1, L] @ values: [N, L, C] -> [N, 1, C]
        weighted_sum = torch.bmm(attention_weights, values) # [N, 1, C]

        # Squeeze to get [N, C] - temporally pooled features
        attended_features = weighted_sum.squeeze(1) # [N, C]

        return attended_features, attention_weights.squeeze(1) # Return features and weights

# --- Main Fusion Network ---
class FusionDomainAdaptationNetwork(nn.Module):
    def __init__(self, handcrafted_input_size, handcrafted_hidden_size, # Handcrafted pre-processing
                 cnn_channels, cnn_kernels, cnn_pools,cnn_input_length, # CNN architecture
                 cnn_attention_dim, # Temporal Attention
                 fusion_interaction_dim, # Interaction layer dim
                 fusion_hidden_size, # Fusion classifier hidden dim
                 dropout_rate,
                 num_classes, # Maturity classes
                 domain_classes, domain_embed_dim, domain_hidden_dim # Domain classifier
                 ):
        super(FusionDomainAdaptationNetwork, self).__init__()
        self.num_classes = num_classes
        self.domain_classes = domain_classes

        # 1. Handcrafted Feature Processor
        # Simple projection for now, could be more complex
        self.handcrafted_processor = nn.Sequential(
            nn.Linear(handcrafted_input_size, handcrafted_hidden_size),
            nn.BatchNorm1d(handcrafted_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
            # Output size: handcrafted_hidden_size
        )
        handcrafted_feature_dim = handcrafted_hidden_size # Use this size for interaction

        # 2. SignalCNN with Residual Blocks
        self.signal_cnn_layers = nn.ModuleList()
        in_channels = 1 # Input is single channel signal
        current_length = -1 # We need cnn_input_length, assumed passed if needed, but attention handles pooling

        for i, (out_channels, kernel_size, pool_size) in enumerate(zip(cnn_channels, cnn_kernels, cnn_pools)):
            # Add residual block
            downsample = None
            stride = 1
            if pool_size > 1: # Use strided convolution for downsampling within the block if pooling is desired conceptually here
                 stride = pool_size # Or keep stride 1 and add separate pooling layer
                 if stride != 1 or in_channels != out_channels:
                      downsample = nn.Sequential(
                           nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                           nn.BatchNorm1d(out_channels)
                      )
            elif in_channels != out_channels: # Handle channel changes without spatial downsampling
                 downsample = nn.Sequential(
                      nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                      nn.BatchNorm1d(out_channels)
                 )

            self.signal_cnn_layers.append(
                ResidualBlock(in_channels, out_channels, kernel_size=kernel_size, stride=stride, downsample=downsample)
            )
            in_channels = out_channels
            # Optional: Add separate MaxPool1d layer *after* the block if stride wasn't used for downsampling
            # if pool_size > 1 and stride == 1:
            #     self.signal_cnn_layers.append(nn.MaxPool1d(kernel_size=pool_size, stride=pool_size))


        # 3. Temporal Attention
        self.temporal_attention = TemporalAttention(feature_dim=in_channels, attention_dim=cnn_attention_dim)
        cnn_feature_dim = in_channels # Feature dim after attention pooling


        # 4. Interaction Layer Components
        self.handcrafted_proj = nn.Linear(handcrafted_feature_dim, fusion_interaction_dim)
        self.cnn_proj = nn.Linear(cnn_feature_dim, fusion_interaction_dim)
        interaction_output_dim = fusion_interaction_dim * 3 # h_proj, c_proj, h_proj * c_proj

        # 5. Fusion Classifier
        self.fusion_fc1 = nn.Linear(interaction_output_dim, fusion_hidden_size)
        self.fusion_bn1 = nn.BatchNorm1d(fusion_hidden_size)
        self.fusion_dropout = nn.Dropout(dropout_rate)
        self.fusion_fc2 = nn.Linear(fusion_hidden_size, num_classes) # Final classification


        # 6. Domain Classifier (using GRL)
        self.domain_grl = GradientReversalLayer()
        self.domain_embedding = nn.Embedding(domain_classes, domain_embed_dim) if domain_embed_dim > 0 else None

        # Input to domain classifier: Use the projected features BEFORE element-wise product for stability? Or use interaction output?
        # Let's use the projected features h_proj, c_proj concatenated.
        domain_input_dim = fusion_interaction_dim * 2 + domain_embed_dim
        self.domain_fc1 = nn.Linear(domain_input_dim, domain_hidden_dim)
        self.domain_bn1 = nn.BatchNorm1d(domain_hidden_dim)
        self.domain_fc2 = nn.Linear(domain_hidden_dim, domain_classes)


    def forward(self, handcrafted, signal, domain_labels=None, lambda_grl=0.0):
        # handcrafted: [N, HandcraftedFeatures]
        # signal: [N, 1, SignalLength]
        # domain_labels: [N] (long tensor)
        # lambda_grl: float

        # 1. Process Handcrafted Features
        h_processed = self.handcrafted_processor(handcrafted) # [N, handcrafted_feature_dim]

        # 2. Process Signal Features (CNN + Attention)
        s = signal
        for layer in self.signal_cnn_layers:
            s = layer(s)
        # s is now [N, C_last, L_last]
        c_attended, attn_weights = self.temporal_attention(s) # [N, cnn_feature_dim]

        # 3. Interaction Layer
        h_proj = self.handcrafted_proj(h_processed) # [N, fusion_interaction_dim]
        c_proj = self.cnn_proj(c_attended)          # [N, fusion_interaction_dim]
        # Apply activation? ReLU or Tanh might be good here before multiplication
        h_proj = torch.relu(h_proj)
        c_proj = torch.relu(c_proj)
        interaction = h_proj * c_proj                # [N, fusion_interaction_dim]
        fused_input = torch.cat((h_proj, c_proj, interaction), dim=1) # [N, fusion_interaction_dim * 3]

        # 4. Fusion Classifier (Predict Maturity Class)
        class_hidden = self.fusion_fc1(fused_input)
        class_hidden = self.fusion_bn1(class_hidden)
        class_hidden = F.relu(class_hidden)
        class_hidden = self.fusion_dropout(class_hidden)
        class_logits = self.fusion_fc2(class_hidden) # [N, num_classes]

        # 5. Domain Classifier (Predict Fruit Type)
        domain_logits = None
        if domain_labels is not None and lambda_grl > 0: # Only compute if needed for training
            # Apply GRL to the features used for domain classification
            # Use the concatenated projections before the interaction product
            domain_features_input = torch.cat((h_proj, c_proj), dim=1) # [N, fusion_interaction_dim * 2]
            domain_features_reversed = self.domain_grl(domain_features_input, lambda_grl) # Apply GRL

            # Concatenate domain embedding if used
            if self.domain_embedding is not None:
                 if domain_labels.dtype != torch.long: domain_labels = domain_labels.long()
                 embeds = self.domain_embedding(domain_labels) # [N, domain_embed_dim]
                 domain_features_combined = torch.cat((domain_features_reversed, embeds), dim=1)
            else:
                 domain_features_combined = domain_features_reversed # No embedding

            # Domain classification layers
            domain_hidden = self.domain_fc1(domain_features_combined)
            domain_hidden = self.domain_bn1(domain_hidden)
            domain_hidden = F.relu(domain_hidden)
            # No dropout usually on domain classifier's last layer? Optional.
            domain_logits = self.domain_fc2(domain_hidden) # [N, domain_classes]
        elif domain_labels is not None and lambda_grl == 0: # Evaluation mode for domain classifier
             # Same path but without GRL
             domain_features_input = torch.cat((h_proj, c_proj), dim=1)
             if self.domain_embedding is not None:
                 if domain_labels.dtype != torch.long: domain_labels = domain_labels.long()
                 embeds = self.domain_embedding(domain_labels)
                 domain_features_combined = torch.cat((domain_features_input, embeds), dim=1)
             else:
                 domain_features_combined = domain_features_input
             domain_hidden = self.domain_fc1(domain_features_combined)
             domain_hidden = self.domain_bn1(domain_hidden)
             domain_hidden = F.relu(domain_hidden)
             domain_logits = self.domain_fc2(domain_hidden)


        # Return class logits, domain logits, and maybe the final fused features used for classification
        final_fused_features_for_vis = class_hidden # Features before final classification layer

        return class_logits, domain_logits, final_fused_features_for_vis