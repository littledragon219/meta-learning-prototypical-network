# model_fusion.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Function

# 设置随机种子
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# 梯度反转层
class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None

class GradientReversalLayer(nn.Module):
    def __init__(self, lambda_=1.0):
        super().__init__()
        self.lambda_ = lambda_
    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)

# 定义 CNN 分支
class SignalCNN(nn.Module):
    def __init__(self, input_length, output_dim):
        super(SignalCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=7, stride=1, padding=3)
        self.bn1 = nn.BatchNorm1d(16)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.flatten = nn.Flatten()
        # 自动计算输出维度
        self._out_dim = self._get_cnn_output_size(input_length)
        self.fc = nn.Linear(self._out_dim, output_dim)
    def _get_cnn_output_size(self, input_length):
        with torch.no_grad():
            x = torch.randn(1, 1, input_length)
            x = self.pool(self.relu(self.bn1(self.conv1(x))))
            x = self.pool2(self.relu(self.bn2(self.conv2(x))))
            x = self.flatten(x)
            return x.size(1)
    def forward(self, x):
        # x: [batch, 1, L]
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu(self.bn2(self.conv2(x))))
        x = self.flatten(x)
        features = self.fc(x)
        return features

# 融合模型：结合手工特征与 CNN 特征（同时用于成熟度分类和水果种类分类）
class FusionDomainAdaptationNetwork(nn.Module):
    def __init__(self, handcrafted_input_size, handcrafted_hidden_size, handcrafted_feature_dim,
                 cnn_input_length, cnn_feature_dim, fusion_hidden_size, dropout_rate,
                 domain_classes, domain_embed_dim, num_classes):
        """
        融合模型：结合手工特征与CNN提取的特征进行分类。
          - 成熟度分类（Label）：num_classes（例如4）；
          - 水果类型分类（Domain）：domain_classes（例如2）。
        """
        super(FusionDomainAdaptationNetwork, self).__init__()
        # 手工特征分支
        self.handcrafted_fc1 = nn.Linear(handcrafted_input_size, handcrafted_hidden_size)
        self.handcrafted_fc2 = nn.Linear(handcrafted_hidden_size, handcrafted_feature_dim)
        self.handcrafted_bn = nn.BatchNorm1d(handcrafted_feature_dim)
        self.handcrafted_dropout = nn.Dropout(dropout_rate)
        
        # CNN 特征分支
        self.signal_cnn = SignalCNN(cnn_input_length, cnn_feature_dim)
        self.cnn_bn = nn.BatchNorm1d(cnn_feature_dim)
        self.cnn_dropout = nn.Dropout(dropout_rate)
        
        # 融合层：将两支特征拼接
        self.fusion_fc1 = nn.Linear(handcrafted_feature_dim + cnn_feature_dim, fusion_hidden_size)
        self.fusion_bn = nn.BatchNorm1d(fusion_hidden_size)
        self.fusion_dropout = nn.Dropout(dropout_rate)
        self.fusion_fc2 = nn.Linear(fusion_hidden_size, num_classes)
        
        # 领域分类器
        self.domain_embedding = nn.Embedding(domain_classes, domain_embed_dim)
        self.domain_fc1 = nn.Linear(handcrafted_feature_dim + cnn_feature_dim, fusion_hidden_size)
        self.domain_grl = GradientReversalLayer()
        self.domain_fc2 = nn.Linear(fusion_hidden_size + domain_embed_dim, domain_classes)
        
    def forward(self, handcrafted, signal, domain_labels=None, lambda_grl=0.0):
        # 手工特征处理
        h = F.relu(self.handcrafted_fc1(handcrafted))
        h = self.handcrafted_fc2(h)
        h = self.handcrafted_bn(h)
        h = self.handcrafted_dropout(h)
        
        # CNN 特征处理
        c = self.signal_cnn(signal)
        c = F.relu(c)
        c = self.cnn_bn(c)
        c = self.cnn_dropout(c)
        
        # 融合两支特征
        fused_feats = torch.cat((h, c), dim=1)
        f = F.relu(self.fusion_fc1(fused_feats))
        f = self.fusion_bn(f)
        f = self.fusion_dropout(f)
        class_logits = self.fusion_fc2(f)
        
        # 领域分类
        if domain_labels is not None:
            self.domain_grl.lambda_ = lambda_grl
            d = self.domain_fc1(fused_feats)
            d = self.domain_grl(d)
            domain_embeds = self.domain_embedding(domain_labels)
            d = torch.cat((d, domain_embeds), dim=1)
            domain_logits = self.domain_fc2(d)
        else:
            domain_logits = None
        
        return class_logits, domain_logits, fused_feats

if __name__ == '__main__':
    # 简单测试
    model = FusionDomainAdaptationNetwork(
        handcrafted_input_size=5,
        handcrafted_hidden_size=128,
        handcrafted_feature_dim=32,
        cnn_input_length=100,
        cnn_feature_dim=32,
        fusion_hidden_size=128,
        dropout_rate=0.2,
        domain_classes=2,
        domain_embed_dim=8,
        num_classes=4
    )
    handcrafted_x = torch.randn(10, 5)
    signal_x = torch.randn(10, 100)
    domain_labels = torch.randint(0, 2, (10,))
    class_logits, domain_logits, fused_feats = model(handcrafted_x, signal_x, domain_labels, lambda_grl=1.0)
    print("class_logits:", class_logits.shape)   # [10, 4]
    print("domain_logits:", domain_logits.shape) # [10, 2]
    print("fused_feats:", fused_feats.shape)
