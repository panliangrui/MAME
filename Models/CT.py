import torch.nn as nn
import numpy as np



# class MLP(nn.Module):
#     def __init__(self, input_dim, num_classes):
#         super(MLP, self).__init__()
#         self.net = nn.Sequential(
#             nn.Linear(input_dim, 64),
#             nn.ReLU(),
#             nn.Linear(64, num_classes)
#         )
#
#     def forward(self, x):
#         return self.net(x)

class MLP(nn.Module):
    def __init__(self, input_dim, num_classes, dropout_rate=0.4):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate / 2),

            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)




import torch
import torch.nn as nn
import torch.nn.functional as F


class FTTransformer(nn.Module):
    def __init__(self,
                 num_features,           # 输入特征数量
                 num_classes=1,          # 输出类别数（=1 表示二分类）
                 d_token=32,             # token embedding size
                 n_heads=4,              # 多头注意力的头数
                 n_blocks=2,             # Transformer Block 数量
                 dropout=0.1,            # dropout 概率
                 use_cls_token=True):    # 是否使用 [CLS] Token
        super().__init__()

        self.use_cls_token = use_cls_token
        self.num_features = num_features
        self.d_token = d_token

        # 将每个数值特征转换为 token
        self.feature_tokenizer = nn.Linear(1, d_token)

        # 可选的 [CLS] token
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_token))

        # Transformer 编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token,
            nhead=n_heads,
            dim_feedforward=4 * d_token,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_blocks)

        # 输出层
        self.head = nn.Sequential(
            nn.LayerNorm(d_token),
            nn.Linear(d_token, num_classes)
        )

    def forward(self, x):
        """
        x: [batch_size, num_features]
        """

        B, F = x.shape

        # 特征 token 化
        x = x.view(B * F, 1)                          # 每个特征独立为一维向量
        x = self.feature_tokenizer(x)                # [B*F, d_token]
        x = x.view(B, F, self.d_token)               # [B, num_features, d_token]

        # 添加 [CLS] token
        if self.use_cls_token:
            cls = self.cls_token.expand(B, -1, -1)   # [B, 1, d_token]
            x = torch.cat([cls, x], dim=1)           # [B, 1+F, d_token]

        # Transformer 编码器
        x = self.transformer(x)                      # [B, 1+F, d_token]

        # 分类输出只取 [CLS] 或平均
        if self.use_cls_token:
            x = x[:, 0, :]  # [CLS] token
        else:
            x = x.mean(dim=1)

        logits = self.head(x)
        return logits

class CNN1D(nn.Module):
    def __init__(self, num_classes):
        super(CNN1D, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = x[:, np.newaxis, :]
        x = self.conv(x)
        x = self.classifier(x)
        return x



class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True)
        self.fc = nn.Linear(hidden_size, 2)
    def forward(self, x):
        if x.dim() == 2:  # 处理形状 [batch, features]
            x = x.unsqueeze(1)  # 变为 [batch, 1, features]
        out, _ = self.lstm(x)  # out: [batch, seq_len, hidden_size]
        return torch.sigmoid(self.fc(out[:, -1, :]))  # 第最后时间步


class FCN(nn.Module):
    def __init__(self, input_dim, num_class):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_class)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc3(x)).squeeze(1)
        return x


import torch
import torch.nn as nn
import math


class MultiHeadAttention(nn.Module):
    """
    多头自注意力机制：能够在多个子空间中学习特征的关系。
    """

    def __init__(self, in_dim=1316, d_model=256, num_heads=8):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        # Query, Key, Value 变换
        self.query = nn.Linear(in_dim, d_model)
        self.key = nn.Linear(in_dim, d_model)
        self.value = nn.Linear(in_dim, d_model)

        # 输出的线性层
        self.fc_out = nn.Linear(d_model, d_model)

    def forward(self, X):
        """
        输入:
        X: [B, F]，输入的特征向量

        输出:
        z: [B, d]，加权后的特征表示
        """
        B = X.size(0)

        # 计算 Q, K, V
        Q = self.query(X)  # [B, d_model]
        K = self.key(X)  # [B, d_model]
        V = self.value(X)  # [B, d_model]

        # 切分为多个头
        Q = Q.view(B, self.num_heads, self.d_model // self.num_heads)
        K = K.view(B, self.num_heads, self.d_model // self.num_heads)
        V = V.view(B, self.num_heads, self.d_model // self.num_heads)

        # 计算注意力得分
        attention_scores = torch.matmul(Q, K.transpose(1, 2)) / math.sqrt(
            self.d_model // self.num_heads)  # [B, heads, F, F]
        attn_weights = torch.softmax(attention_scores, dim=-1)

        # 计算加权求和的特征表示
        z = torch.matmul(attn_weights, V)  # [B, heads, F, d_model // heads]
        z = z.view(B, -1)  # 合并多个头的输出

        # 输出的线性变换
        z = self.fc_out(z)  # [B, d_model]

        return z


class CTBranch(nn.Module):
    def __init__(self, in_dim=1316, d=256, num_heads=8, num_classes=2, dropout=0.2):
        super().__init__()
        self.attention = MultiHeadAttention(in_dim=in_dim, d_model=d, num_heads=num_heads)
        self.out_proj = nn.Linear(d, d)  # 最后映射到 d 维
        self.layer_norm = nn.LayerNorm(d)
        self.fc = nn.Linear(d, num_classes)  # 输出层进行二分类
        self.sigmoid = nn.Sigmoid()  # 用于二分类的sigmoid激活函数
        self.dropout = nn.Dropout(dropout)  # Dropout 层

        # 为了将输入 X 映射到与 z 相同的维度，我们加一个额外的线性层
        self.input_proj = nn.Linear(in_dim, d)  # 将输入 X 映射到 d 维

    def forward(self, X):
        """
        输入:
        X: [B, 1316] 或 [1, 1316]，CT 特征

        输出:
        z: [B, 2] 或 [1, 2]，加权后的特征表示（二分类）
        """
        if X.dim() == 1:
            X = X.unsqueeze(0)  # [1, F]，如果只有一条样本

        # 通过多头注意力机制提取特征
        z = self.attention(X)  # z: [B, d]

        # 映射 X 到与 z 相同的维度
        X_proj = self.input_proj(X)  # [B, d]

        # 残差连接：将映射后的 X 和 z 相加
        z = z + X_proj  # 残差连接

        # Dropout 层：防止过拟合
        z = self.dropout(z)

        # 最后的输出，映射到指定的维度
        z = self.out_proj(z)  # [B, d]
        z = self.layer_norm(z)

        # 二分类输出，使用 sigmoid 计算类别概率
        z = self.fc(z)  # [B, num_classes]，num_classes = 2
        # z = self.sigmoid(z)  # 使用 sigmoid 进行二分类

        return z

# class CTBranch(nn.Module):
#     def __init__(self, in_dim=1316, d=256, num_classes=2):
#         super().__init__()
#         self.fc1 = nn.Linear(in_dim, d)  # 输入层
#         self.fc2 = nn.Linear(d, d)       # 隐藏层
#         self.fc3 = nn.Linear(d, num_classes)  # 输出层
#         self.relu = nn.ReLU()
#         self.layer_norm = nn.LayerNorm(d)
#
#     def forward(self, X):
#         """
#         输入:
#         X: [B, 1316] 或 [1, 1316]，CT 特征
#
#         输出:
#         z: [B, 2] 或 [1, 2]，加权后的特征表示（二分类）
#         """
#         if X.dim() == 1:
#             X = X.unsqueeze(0)  # [1, F]，如果只有一条样本
#
#         # 通过全连接层进行特征学习
#         z = self.fc1(X)  # [B, d]
#         z = self.relu(z)
#         z = self.fc2(z)  # [B, d]
#         z = self.relu(z)
#         z = self.layer_norm(z)  # LayerNorm 用于标准化
#         z = self.fc3(z)  # [B, num_classes]
#         return z





