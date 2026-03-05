import os
from torch_scatter import scatter_add
from torch_geometric.utils import softmax
from torch_geometric.nn import SAGEConv, LayerNorm
import torch

from torch_geometric.nn import GATConv, GCNConv, GENConv, GINConv, GMMConv, GPSConv, GINEConv, GATv2Conv, APPNP, \
    GatedGraphConv, ARMAConv

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import math, os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor, matmul
from torch_geometric.utils import degree


def full_attention_conv(qs, ks, vs, kernel, output_attn=False):
    '''
    qs: query tensor [N, H, M]
    ks: key tensor [L, H, M]
    vs: value tensor [L, H, D]

    return output [N, H, D]
    '''
    if kernel == 'simple':
        # normalize input
        qs = qs / torch.norm(qs, p=2)  # [N, H, M]
        ks = ks / torch.norm(ks, p=2)  # [L, H, M]
        N = qs.shape[0]

        # numerator
        kvs = torch.einsum("lhm,lhd->hmd", ks, vs)
        attention_num = torch.einsum("nhm,hmd->nhd", qs, kvs)  # [N, H, D]
        all_ones = torch.ones([vs.shape[0]]).to(vs.device)
        vs_sum = torch.einsum("l,lhd->hd", all_ones, vs)  # [H, D]
        attention_num += vs_sum.unsqueeze(0).repeat(vs.shape[0], 1, 1)  # [N, H, D]

        # denominator
        all_ones = torch.ones([ks.shape[0]]).to(ks.device)
        ks_sum = torch.einsum("lhm,l->hm", ks, all_ones)
        attention_normalizer = torch.einsum("nhm,hm->nh", qs, ks_sum)  # [N, H]

        # attentive aggregated results
        attention_normalizer = torch.unsqueeze(attention_normalizer, len(attention_normalizer.shape))  # [N, H, 1]
        attention_normalizer += torch.ones_like(attention_normalizer) * N
        attn_output = attention_num / attention_normalizer  # [N, H, D]

        # compute attention for visualization if needed
        if output_attn:
            attention = torch.einsum("nhm,lhm->nlh", qs, ks) / attention_normalizer  # [N, L, H]

    elif kernel == 'sigmoid':
        # numerator
        attention_num = torch.sigmoid(torch.einsum("nhm,lhm->nlh", qs, ks))  # [N, L, H]

        # denominator
        all_ones = torch.ones([ks.shape[0]]).to(ks.device)
        attention_normalizer = torch.einsum("nlh,l->nh", attention_num, all_ones)
        attention_normalizer = attention_normalizer.unsqueeze(1).repeat(1, ks.shape[0], 1)  # [N, L, H]

        # compute attention and attentive aggregated results
        attention = attention_num / attention_normalizer
        attn_output = torch.einsum("nlh,lhd->nhd", attention, vs)  # [N, H, D]

    if output_attn:
        return attn_output, attention
    else:
        return attn_output


def gcn_conv(x, edge_index, edge_weight):
    N, H = x.shape[0], x.shape[1]
    row, col = edge_index
    d = degree(col, N).float()
    d_norm_in = (1. / d[col]).sqrt()
    d_norm_out = (1. / d[row]).sqrt()
    gcn_conv_output = []
    if edge_weight is None:
        value = torch.ones_like(row) * d_norm_in * d_norm_out
    else:
        value = edge_weight * d_norm_in * d_norm_out
    value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
    adj = SparseTensor(row=col, col=row, value=value, sparse_sizes=(N, N))
    for i in range(x.shape[1]):
        gcn_conv_output.append(matmul(adj, x[:, i]))  # [N, D]
    gcn_conv_output = torch.stack(gcn_conv_output, dim=1)  # [N, H, D]
    return gcn_conv_output


class DIFFormerConv(nn.Module):
    '''
    one DIFFormer layer
    '''

    def __init__(self, in_channels,
                 out_channels,
                 num_heads,
                 kernel='simple',
                 use_graph=True,
                 use_weight=True):
        super(DIFFormerConv, self).__init__()
        self.Wk = nn.Linear(in_channels, out_channels * num_heads)
        self.Wq = nn.Linear(in_channels, out_channels * num_heads)
        if use_weight:
            self.Wv = nn.Linear(in_channels, out_channels * num_heads)

        self.out_channels = out_channels
        self.num_heads = num_heads
        self.kernel = kernel
        self.use_graph = use_graph
        self.use_weight = use_weight

    def reset_parameters(self):
        self.Wk.reset_parameters()
        self.Wq.reset_parameters()
        if self.use_weight:
            self.Wv.reset_parameters()

    def forward(self, query_input, source_input, edge_index=None, edge_weight=None, output_attn=False):
        # feature transformation
        query = self.Wq(query_input).reshape(-1, self.num_heads, self.out_channels)
        key = self.Wk(source_input).reshape(-1, self.num_heads, self.out_channels)
        if self.use_weight:
            value = self.Wv(source_input).reshape(-1, self.num_heads, self.out_channels)
        else:
            value = source_input.reshape(-1, 1, self.out_channels)

        # compute full attentive aggregation
        if output_attn:
            attention_output, attn = full_attention_conv(query, key, value, self.kernel, output_attn)  # [N, H, D]
        else:
            attention_output = full_attention_conv(query, key, value, self.kernel)  # [N, H, D]

        # use input graph for gcn conv
        if self.use_graph:
            final_output = attention_output + gcn_conv(value, edge_index, edge_weight)
        else:
            final_output = attention_output
        final_output = final_output.mean(dim=1)

        if output_attn:
            return final_output, attn
        else:
            return final_output


class DIFFormer(nn.Module):
    '''
    DIFFormer model class
    x: input node features [N, D]
    edge_index: 2-dim indices of edges [2, E]
    return y_hat predicted logits [N, C]
    '''

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=1, num_heads=1, kernel='simple',
                 alpha=0.5, dropout=0.2, use_bn=True, use_residual=True, use_weight=True, use_graph=True):
        super(DIFFormer, self).__init__()

        self.convs = nn.ModuleList()
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(in_channels, hidden_channels))
        self.bns = nn.ModuleList()
        self.bns.append(nn.LayerNorm(hidden_channels))
        for i in range(num_layers):
            self.convs.append(
                DIFFormerConv(hidden_channels, hidden_channels, num_heads=num_heads, kernel=kernel, use_graph=use_graph,
                              use_weight=use_weight))
            self.bns.append(nn.LayerNorm(hidden_channels))

        self.fcs.append(nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout
        self.activation = F.relu
        self.use_bn = use_bn
        self.residual = use_residual
        self.alpha = alpha

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        for fc in self.fcs:
            fc.reset_parameters()

    def forward(self, x, edge_index, edge_weight=None):
        layer_ = []

        # input MLP layer
        x = self.fcs[0](x)
        if self.use_bn:
            x = self.bns[0](x)
        x = self.activation(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # store as residual link
        layer_.append(x)

        for i, conv in enumerate(self.convs):
            # graph convolution with DIFFormer layer
            x = conv(x, x, edge_index, edge_weight)
            if self.residual:
                x = self.alpha * x + (1 - self.alpha) * layer_[i]
            if self.use_bn:
                x = self.bns[i + 1](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            layer_.append(x)

        # output MLP layer
        x_out = self.fcs[-1](x)
        return x_out

    def get_attentions(self, x):
        layer_, attentions = [], []
        x = self.fcs[0](x)
        if self.use_bn:
            x = self.bns[0](x)
        x = self.activation(x)
        layer_.append(x)
        for i, conv in enumerate(self.convs):
            x, attn = conv(x, x, output_attn=True)
            attentions.append(attn)
            if self.residual:
                x = self.alpha * x + (1 - self.alpha) * layer_[i]
            if self.use_bn:
                x = self.bns[i + 1](x)
            layer_.append(x)
        return torch.stack(attentions, dim=0)  # [layer num, N, N]


import torch.nn as nn
from torch_sparse import SparseTensor, matmul
from torch_geometric.utils import degree


class GraphConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, use_weight=True, use_init=False):
        super(GraphConvLayer, self).__init__()

        self.use_init = use_init
        self.use_weight = use_weight
        if self.use_init:
            in_channels_ = 2 * in_channels
        else:
            in_channels_ = in_channels
        self.W = nn.Linear(in_channels_, out_channels)

    def reset_parameters(self):
        self.W.reset_parameters()

    def forward(self, x, edge_index, x0):
        N = x.shape[0]
        row, col = edge_index
        d = degree(col, N).float()
        d_norm_in = (1. / d[col]).sqrt()
        d_norm_out = (1. / d[row]).sqrt()
        value = torch.ones_like(row) * d_norm_in * d_norm_out
        value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
        adj = SparseTensor(row=col, col=row, value=value, sparse_sizes=(N, N))
        x = matmul(adj, x)  # [N, D]

        if self.use_init:
            x = torch.cat([x, x0], 1)
            x = self.W(x)
        elif self.use_weight:
            x = self.W(x)

        return x


class GraphConv(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers=2, dropout=0.5, use_bn=True, use_residual=True,
                 use_weight=True, use_init=False, use_act=True):
        super(GraphConv, self).__init__()

        self.convs = nn.ModuleList()
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(in_channels, hidden_channels))

        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers):
            self.convs.append(
                GraphConvLayer(hidden_channels, hidden_channels, use_weight, use_init))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        self.dropout = dropout
        self.activation = F.relu
        self.use_bn = use_bn
        self.use_residual = use_residual
        self.use_act = use_act

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        for fc in self.fcs:
            fc.reset_parameters()

    def forward(self, x, edge_index):
        layer_ = []

        x = self.fcs[0](x)
        if self.use_bn:
            x = self.bns[0](x)
        x = self.activation(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        layer_.append(x)

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, layer_[0])
            if self.use_bn:
                x = self.bns[i + 1](x)
            if self.use_act:
                x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if self.use_residual:
                x = x + layer_[-1]
        return x


class TransConvLayer(nn.Module):
    '''
    transformer with fast attention
    '''

    def __init__(self, in_channels,
                 out_channels,
                 num_heads,
                 use_weight=True):
        super().__init__()
        self.Wk = nn.Linear(in_channels, out_channels * num_heads)
        self.Wq = nn.Linear(in_channels, out_channels * num_heads)
        if use_weight:
            self.Wv = nn.Linear(in_channels, out_channels * num_heads)

        self.out_channels = out_channels
        self.num_heads = num_heads
        self.use_weight = use_weight

    def reset_parameters(self):
        self.Wk.reset_parameters()
        self.Wq.reset_parameters()
        if self.use_weight:
            self.Wv.reset_parameters()

    def forward(self, query_input, source_input, output_attn=False):
        # feature transformation
        qs = self.Wq(query_input).reshape(-1, self.num_heads, self.out_channels)
        ks = self.Wk(source_input).reshape(-1, self.num_heads, self.out_channels)
        if self.use_weight:
            vs = self.Wv(source_input).reshape(-1, self.num_heads, self.out_channels)
        else:
            vs = source_input.reshape(-1, 1, self.out_channels)

        # normalize input
        qs = qs / torch.norm(qs, p=2)  # [N, H, M]
        ks = ks / torch.norm(ks, p=2)  # [L, H, M]
        N = qs.shape[0]

        # numerator
        kvs = torch.einsum("lhm,lhd->hmd", ks, vs)
        attention_num = torch.einsum("nhm,hmd->nhd", qs, kvs)  # [N, H, D]
        attention_num += N * vs

        # denominator
        all_ones = torch.ones([ks.shape[0]]).to(ks.device)
        ks_sum = torch.einsum("lhm,l->hm", ks, all_ones)
        attention_normalizer = torch.einsum("nhm,hm->nh", qs, ks_sum)  # [N, H]

        # attentive aggregated results
        attention_normalizer = torch.unsqueeze(
            attention_normalizer, len(attention_normalizer.shape))  # [N, H, 1]
        attention_normalizer += torch.ones_like(attention_normalizer) * N
        attn_output = attention_num / attention_normalizer  # [N, H, D]

        # compute attention for visualization if needed
        if output_attn:
            attention = torch.einsum("nhm,lhm->nlh", qs, ks).mean(dim=-1)  # [N, N]
            normalizer = attention_normalizer.squeeze(dim=-1).mean(dim=-1, keepdims=True)  # [N,1]
            attention = attention / normalizer

        final_output = attn_output.mean(dim=1)

        if output_attn:
            return final_output, attention
        else:
            return final_output


class TransConv(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers=2, num_heads=1,
                 dropout=0.2, use_bn=True, use_residual=True, use_weight=True, use_act=True):
        super().__init__()

        self.convs = nn.ModuleList()
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(in_channels, hidden_channels))
        self.bns = nn.ModuleList()
        self.bns.append(nn.LayerNorm(hidden_channels))
        for i in range(num_layers):
            self.convs.append(
                DIFFormerConv(hidden_channels, hidden_channels, num_heads=num_heads, kernel='simple', use_graph=False,
                              use_weight=use_weight))
            # TransConvLayer(hidden_channels, hidden_channels, num_heads=num_heads, use_weight=use_weight))
            self.bns.append(nn.LayerNorm(hidden_channels))

        self.dropout = dropout
        self.activation = F.relu
        self.use_bn = use_bn
        self.use_residual = use_residual
        self.use_act = use_act

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        for fc in self.fcs:
            fc.reset_parameters()

    def forward(self, x):
        layer_ = []

        # input MLP layer
        x = self.fcs[0](x)
        if self.use_bn:
            x = self.bns[0](x)
        x = self.activation(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # store as residual link
        layer_.append(x)

        for i, conv in enumerate(self.convs):
            # graph convolution with full attention aggregation
            x = conv(x, x)
            if self.use_residual:
                x = (x + layer_[i]) / 2.
            if self.use_bn:
                x = self.bns[i + 1](x)
            if self.use_act:
                x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            layer_.append(x)

        return x

    def get_attentions(self, x):
        layer_, attentions = [], []
        x = self.fcs[0](x)
        if self.use_bn:
            x = self.bns[0](x)
        x = self.activation(x)
        layer_.append(x)
        for i, conv in enumerate(self.convs):
            x, attn = conv(x, x, output_attn=True)
            attentions.append(attn)
            if self.use_residual:
                x = (x + layer_[i]) / 2.
            if self.use_bn:
                x = self.bns[i + 1](x)
            if self.use_act:
                x = self.activation(x)
            layer_.append(x)
        return torch.stack(attentions, dim=0)  # [layer num, N, N]


# 定义一个使用 Transformer 自注意力机制的模块，添加输入投影层
class TransformerAttention(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads):
        super(TransformerAttention, self).__init__()
        self.input_proj = nn.Linear(input_dim, embed_dim)  # 输入投影层
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)
        # 可选：添加前馈网络
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, 400),
            nn.ReLU(),
            nn.Linear(400, embed_dim)
        )

    def forward(self, x):
        # x 的形状: (batch_size, seq_len, input_dim)
        x = self.input_proj(x)  # 投影到固定的 embed_dim
        x = x.permute(1, 0, 2)  # (seq_len, batch_size, embed_dim)

        # 自注意力机制
        attn_output, attn_weights = self.attention(x, x, x)

        # 可选：通过前馈网络
        # attn_output = self.feed_forward(attn_output)

        # 降维：对 seq_len 维度进行平均
        context = attn_output.mean(dim=0)  # (batch_size, embed_dim)

        return context, attn_weights


# from Models.SAT import GraphTransformer
# from Models.difformer import DIFFormer
# from Models.CoBFormer import CoBFormer
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool


# from torch_geometric.data import Data
# from Models.vit_one import TransformerFeatureExtractor, CompleteModel
# from Models.mamba import GPSConv
# from mamba_ssm.modules.mamba_simple import Mamba
# from torch.nn import (
#     BatchNorm1d,
#     Embedding,
#     Linear,
#     ModuleList,
#     ReLU,
#     Sequential,
# )
class GCN_TME(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, graph_weight=0.8):
        super(GCN_TME, self).__init__()
        self.graph_weight = graph_weight
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        # self.fc = torch.nn.Linear(hidden_channels, out_channels)
        # 全连接层用于分类
        self.fc1 = torch.nn.Linear(hidden_channels * 2, 8)
        self.fc2 = torch.nn.Linear(8, out_channels)  # 假设二分类
        self.ffn_norm2_1 = LayerNorm(in_channels=1, eps=1e-5)

    def forward(self, node_features, edges):
        x, edge_index = node_features, edges  # data.node, data.edge_index
        x_r = x

        # 第一层卷积
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)

        x_r = self.conv1(x_r, edge_index)
        x_r = F.relu(x_r)
        x_r = F.dropout(x_r, p=0.2, training=self.training)

        # if self.aggregate == 'add':
        x = self.graph_weight * x + (1 - self.graph_weight) * x_r

        # 第二层卷积
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.ffn_norm2_1(x)

        # # 全局池化，将每个节点的特征聚合成一个全图特征
        # x = global_mean_pool(x, data.batch)  # 假设 `data.batch` 存在，否则默认图是一个整体
        #
        # # 全连接层进行图分类
        # x = self.fc(x)

        # # 使用全局池化提取全局特征
        # global_mean_x = global_mean_pool(x, batch=None)  # 全局均值池化
        # global_max_x = global_max_pool(x, batch=None)  # 全局最大池化
        #
        # # 将全局特征拼接
        # global_x = torch.cat([global_mean_x, global_max_x], dim=1)

        # 全连接层进行分类
        # x = self.fc1(global_x)
        # x = F.relu(x)
        # x = self.fc2(x)

        return x  # return F.log_softmax(x, dim=1)


# from xformers.ops import memory_efficient_attention
# class FlashAttentionFeatureExtractor(nn.Module):
#     def __init__(self, input_dim=768, output_dim=400, num_heads=8):
#         super(FlashAttentionFeatureExtractor, self).__init__()
#         self.num_heads = num_heads
#         self.head_dim = input_dim // num_heads  # 每个头的嵌入维度
#
#         # 多头注意力的线性投影层
#         self.query_proj = nn.Linear(input_dim, input_dim)
#         self.key_proj = nn.Linear(input_dim, input_dim)
#         self.value_proj = nn.Linear(input_dim, input_dim)
#
#         # 特征降维的线性层 (将768维降到400维)
#         self.feature_proj = nn.Linear(input_dim, output_dim)
#
#     def forward(self, x):
#         # x 的形状为 (50000, 768)
#         x = x.unsqueeze(0)  # 添加 batch 维度，变为 (1, 50000, 768)
#
#         # 投影并拆分为多个头
#         queries = self.query_proj(x).view(1, -1, self.num_heads, self.head_dim)  # (1, 50000, num_heads, head_dim)
#         keys = self.key_proj(x).view(1, -1, self.num_heads, self.head_dim)
#         values = self.value_proj(x).view(1, -1, self.num_heads, self.head_dim)
#
#         # 使用 Flash Attention 计算
#         attn_output = memory_efficient_attention(queries, keys, values)  # (1, 50000, num_heads, head_dim)
#
#         # 合并头部输出
#         attn_output = attn_output.view(1, -1, self.num_heads * self.head_dim)  # (1, 50000, 768)
#
#         # 去除 batch 维度，回到 (50000, 768)
#         attn_output = attn_output.squeeze(0)
#
#         # 特征降维，将768维降到400维
#         reduced_features = self.feature_proj(attn_output)  # (50000, 400)
#
#         return reduced_features


class FixedOutputModel(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=2048, output_dim=1024):
        super(FixedOutputModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.pooling = nn.AdaptiveMaxPool1d(hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x.shape: (batch_size, seq_len, input_dim)
        # LSTM
        # lstm_out, _ = self.lstm(x)
        # lstm_out.shape: (batch_size, seq_len, hidden_dim)

        # Pooling
        # 需要调整维度以匹配池化层期望的输入 (batch_size, channels, seq_len)
        # pooled = self.pooling(lstm_out.transpose(1, 2))
        pooled = self.pooling(x)
        # pooled.shape: (batch_size, hidden_dim, 1)

        # Flatten the output for the linear layer
        flattened = pooled.squeeze(-1)
        # flattened.shape: (batch_size, hidden_dim)

        # Fully connected layer to ensure the output dimension is fixed at 1024
        # output = self.fc(flattened)
        # output.shape: (batch_size, output_dim)

        return flattened  # output


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1024):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维度
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维度
        pe = pe.unsqueeze(1)  # (max_len, 1, d_model)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (seq_len, batch_size * num_chunks, embed_dim)
        x = x + self.pe[:x.size(0), :]
        return x


class TransformerFeatureExtractor(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, num_layers, chunk_size):
        super(TransformerFeatureExtractor, self).__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.chunk_size = chunk_size

        # 输入投影层，将 input_dim 映射到 embed_dim
        self.input_proj = nn.Linear(input_dim, embed_dim)

        # 位置编码
        self.positional_encoding = PositionalEncoding(embed_dim, max_len=chunk_size)

        # Transformer 编码器层
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 池化层，将序列特征聚合为单个向量
        self.pooling = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = x.unsqueeze(2)
        # x: (batch_size, seq_len, input_dim)
        batch_size, seq_len, _ = x.size()

        # 检查序列长度是否能被 chunk_size 整除，不能则进行填充
        if seq_len % self.chunk_size != 0:
            pad_size = self.chunk_size - (seq_len % self.chunk_size)
            x = F.pad(x.transpose(1, 2), (0, pad_size), "constant", 0).transpose(1, 2)
            seq_len = x.size(1)

        # 计算块的数量
        num_chunks = seq_len // self.chunk_size

        # 将序列划分为多个块
        x = x.view(batch_size * num_chunks, self.chunk_size, self.input_dim)

        # 输入投影
        x = self.input_proj(x)  # (batch_size * num_chunks, chunk_size, embed_dim)

        # 转置为 (chunk_size, batch_size * num_chunks, embed_dim)
        x = x.transpose(0, 1)

        # 添加位置编码
        x = self.positional_encoding(x)

        # 通过 Transformer 编码器
        x = self.transformer_encoder(x)

        # 转置回 (batch_size * num_chunks, chunk_size, embed_dim)
        x = x.transpose(0, 1)

        # 池化层，得到每个块的特征表示
        x = x.transpose(1, 2)  # (batch_size * num_chunks, embed_dim, chunk_size)
        x = self.pooling(x)  # (batch_size * num_chunks, embed_dim, 1)
        x = x.squeeze(-1)  # (batch_size * num_chunks, embed_dim)

        # 将块的特征重新组合为 (batch_size, num_chunks, embed_dim)
        x = x.view(batch_size, num_chunks, self.embed_dim)

        # 对所有块的特征进行全局池化（例如平均池化）
        x = x.mean(dim=1)  # (batch_size, embed_dim)

        return x


class AdaptiveFeatureFusion(nn.Module):
    def __init__(self, target_dim=1024):
        super(AdaptiveFeatureFusion, self).__init__()
        # 定义全连接层将每个特征映射到相同维度
        self.fc1 = nn.Linear(1024, target_dim)
        self.fc2 = nn.Linear(2048, target_dim)
        self.fc3 = nn.Linear(1024, target_dim)
        self.fc4 = nn.Linear(1024, target_dim)
        # self.mamba_model1 = Mamba(d_model=1, d_state=256, d_conv=4, expand=2).to("cuda")
        # self.mamba_model2 = Mamba(d_model=1, d_state=128, d_conv=4, expand=2).to("cuda")
        # self.mamba_model3 = Mamba(d_model=1, d_state=64, d_conv=4, expand=2).to("cuda")
        # self.mamba_model4 = Mamba(d_model=1, d_state=64, d_conv=4, expand=2).to("cuda")
        self.atten = TransformerFeatureExtractor(input_dim=1, embed_dim=4608, num_heads=8, num_layers=1, chunk_size=32)
        self.fc5 = nn.Linear(4608, target_dim)

    def forward(self, x1, x2, x3, x4):
        x1 = self.fc1(x1)
        x2 = self.fc1(x2)
        concatenated_tensor = torch.cat((x1, x2, x3, x4), dim=1)
        # 将各特征映射到相同维度
        # out1 = self.fc1(x1)
        # out2 = self.fc2(x2)
        # out3 = self.fc3(x3)
        # out4 = self.fc4(x4)
        x1_2 = concatenated_tensor  # .unsqueeze(2)
        # x1_1 = x1.unsqueeze(1)
        # out1 = self.mamba_model1(x1_2)
        out1 = self.atten(x1_2)
        fused_features = self.fc5(out1)
        return fused_features


class SEBlock(nn.Module):
    def __init__(self, input_dim, reduction_ratio=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(input_dim, input_dim // reduction_ratio, bias=False)
        self.fc2 = nn.Linear(input_dim // reduction_ratio, input_dim, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 全局平均池化
        # out = torch.mean(x, dim=1, keepdim=True)
        # 两层全连接层
        out = self.fc1(x)
        out = self.fc2(out)
        # Sigmoid激活获取权重
        weight = self.sigmoid(out)
        # 权重与输入特征相乘
        out = x * weight
        return out


class Classifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


class CompleteModel(nn.Module):
    def __init__(self, target_dim=1024, num_classes=10):
        super(CompleteModel, self).__init__()
        # 特征融合模块
        self.fusion = AdaptiveFeatureFusion(target_dim)
        # 自适应权重模块
        self.se = SEBlock(target_dim)  # 融合后的维度为4倍的target_dim
        # 分类器
        self.classifier = Classifier(target_dim, num_classes)

    def forward(self, x1, x2, x3, x4):
        # 融合特征
        fused_features = self.fusion(x1, x2, x3, x4)
        # 通过自适应模块
        weighted_features = self.se(fused_features)
        # 进行分类
        output = self.classifier(weighted_features)
        return output


from Models.difformer import full_attention_conv


class attenN(nn.Module):
    '''
    atten N
    '''

    def __init__(self, in_channels,
                 out_channels,
                 num_heads,
                 kernel='simple',
                 use_weight=True):
        super(attenN, self).__init__()
        self.Wk = nn.Linear(in_channels, out_channels * num_heads)
        self.Wq = nn.Linear(in_channels, out_channels * num_heads)
        if use_weight:
            self.Wv = nn.Linear(in_channels, out_channels * num_heads)

        self.out_channels = out_channels
        self.num_heads = num_heads
        self.kernel = kernel
        # self.use_graph = use_graph
        self.use_weight = use_weight

    def reset_parameters(self):
        self.Wk.reset_parameters()
        self.Wq.reset_parameters()
        if self.use_weight:
            self.Wv.reset_parameters()

    def forward(self, query_input, source_input, output_attn=False):
        # feature transformation
        query = self.Wq(query_input).reshape(-1, self.num_heads, self.out_channels)
        key = self.Wk(source_input).reshape(-1, self.num_heads, self.out_channels)
        if self.use_weight:
            value = self.Wv(source_input).reshape(-1, self.num_heads, self.out_channels)
        else:
            value = source_input.reshape(-1, 1, self.out_channels)

        # compute full attentive aggregation
        if output_attn:
            attention_output, attn = full_attention_conv(query, key, value, self.kernel, output_attn)  # [N, H, D]
        else:
            attention_output = full_attention_conv(query, key, value, self.kernel)  # [N, H, D]

        # use input graph for gcn conv
        # if self.use_graph:
        #     final_output = attention_output + gcn_conv(value, edge_index, edge_weight)
        # else:
        final_output = attention_output
        final_output = final_output.mean(dim=1)

        if output_attn:
            return final_output, attn
        else:
            return final_output


def GNN_relu_Block(dim2, dropout=0.1):
    r"""
    Multilayer Reception Block w/ Self-Normalization (Linear + ELU + Alpha Dropout)
    args:
        dim1 (int): Dimension of input features
        dim2 (int): Dimension of output features
        dropout (float): Dropout rate
    """
    return nn.Sequential(
        nn.ReLU(),
        LayerNorm(dim2),
        nn.Dropout(p=dropout))


class fusion_model_graph(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 trans_num_layers=1, trans_num_heads=8, trans_dropout=0.2, trans_use_bn=True, trans_use_residual=True,
                 trans_use_weight=True, trans_use_act=True,
                 gnn_num_layers=1, gnn_dropout=0.2, gnn_use_weight=True, gnn_use_init=False, gnn_use_bn=True,
                 gnn_use_residual=True, gnn_use_act=True,
                 use_graph=True, graph_weight=0.8, aggregate='add'):
        super().__init__()

        # self.trans_conv = TransConv(in_channels, hidden_channels, trans_num_layers, trans_num_heads, trans_dropout,
        #                             trans_use_bn, trans_use_residual, trans_use_weight, trans_use_act)
        # #FlashAttentionFeatureExtractor(input_dim=in_channels, output_dim=hidden_channels, num_heads=trans_num_heads)
        # self.graph_conv = GraphConv(in_channels, hidden_channels, gnn_num_layers, gnn_dropout, gnn_use_bn,
        #                             gnn_use_residual, gnn_use_weight, gnn_use_init, gnn_use_act)
        # self.use_graph = use_graph
        # self.graph_weight = graph_weight
        #
        # self.aggregate = aggregate
        #
        # if aggregate == 'add':
        #     self.fc = nn.Linear(hidden_channels, 200)#300)#out_channels
        # elif aggregate == 'cat':
        #     self.fc = nn.Linear(2 * hidden_channels, 200)#300)#out_channels
        # else:
        #     raise ValueError(f'Invalid aggregate type:{aggregate}')
        #
        # self.params1 = list(self.trans_conv.parameters())
        # self.params2 = list(self.graph_conv.parameters()) if self.graph_conv is not None else []
        # self.params2.extend(list(self.fc.parameters()))

        ##对于TME特征提取
        # self.graph_transformer = GraphTransformer(in_size= 1, num_class=out_channels , d_model=8)
        # self.gcn_tme = GCN_TME(in_channels=11, hidden_channels=2, out_channels=1)
        # ###difformer
        # self.difformer = DIFFormer(in_channels=768, hidden_channels=40, out_channels=1)
        self.img_gnn1 = SAGEConv(in_channels=768, out_channels=1)
        self.img_gnn2 = SAGEConv(in_channels=11, out_channels=1)
        self.gnn_relu = GNN_relu_Block(dim2=1, dropout=0.2)

        # self.attenNN = attenN(hidden_channels, hidden_channels, num_heads=1, kernel='simple',  use_weight=True)
        ##特征融合模块
        # y = model(x)
        self.attention_1024 = self.pooling = nn.AdaptiveMaxPool1d(
            512)  # FixedOutputModel(input_dim=1, hidden_dim=2048, output_dim=128)#TransformerFeatureExtractor(input_dim=1, embed_dim=256, num_heads=8, num_layers=1, chunk_size=256)
        self.attention_512 = self.pooling = nn.AdaptiveMaxPool1d(
            1024)  # FixedOutputModel(input_dim=1, hidden_dim=2048, output_dim=256)#TransformerFeatureExtractor(input_dim=1, embed_dim=512, num_heads=8, num_layers=1, chunk_size=256)#TransformerAttention(input_dim=1, embed_dim=800, num_heads=8)
        self.attention_256 = self.pooling = nn.AdaptiveMaxPool1d(
            2048)  # FixedOutputModel(input_dim=1, hidden_dim=2048, output_dim=512)#TransformerFeatureExtractor(input_dim=1, embed_dim=1024, num_heads=8, num_layers=1, chunk_size=256)#TransformerAttention(input_dim=1, embed_dim=1200, num_heads=8)
        self.attention_TME = self.pooling = nn.AdaptiveMaxPool1d(
            1024)  # FixedOutputModel(input_dim=1, hidden_dim=2048, output_dim=1024)

        self.atten = TransformerFeatureExtractor(input_dim=1, embed_dim=1024, num_heads=8, num_layers=1,
                                                 chunk_size=32)  # 1024
        self.atten1 = DIFFormerConv(in_channels=3072, out_channels=1024, num_heads=1, kernel='simple', use_graph=False,
                                    use_weight=True)  # 5120 3072
        self.atten2 = DIFFormerConv(in_channels=2048, out_channels=1024, num_heads=1, kernel='simple', use_graph=False,
                                    use_weight=True)  # 3072   2048

        self.adapt_module = CompleteModel(target_dim=1024, num_classes=out_channels)
        self.fc1 = nn.Linear(1024, out_channels)
        self.fc2 = nn.Linear(1024, out_channels)
        self.fc3 = nn.Linear(512, out_channels)
        self.fc4 = nn.Linear(1024, out_channels)

    def forward(self, node_image_path_256_fea, node_image_path_512_fea, node_image_path_1024_fea, edge_index_image_256,
                edge_index_image_512, edge_index_image_1024, node_features,
                edges):  # node_image_path_256_fea,node_image_path_512_fea,node_image_path_1024_fea,edge_index_image_256,edge_index_image_512,edge_index_image_1024,bag_feats_TME.node,bag_feats_TME.edge_index
        ##SGFormer get features from 20X
        # x_256_1 = self.trans_conv(node_image_path_256_fea)
        # if self.use_graph:
        #     x_256_2 = self.graph_conv(node_image_path_256_fea, edge_index_image_256)
        #     if self.aggregate == 'add':
        #         x_256 = self.graph_weight * x_256_2 + (1 - self.graph_weight) * x_256_1
        #     else:
        #         x_256 = torch.cat((x_256_1, x_256_2), dim=1)
        # else:
        #     x_256 = x_256_1
        # x_256 = self.fc(x_256)
        # ###graph_difformer
        # x_256_d = self.difformer(x_256, edge_index_image_256)
        x_256_d = self.img_gnn1(node_image_path_256_fea, edge_index_image_256)
        x_256_d = self.gnn_relu(x_256_d)

        ##SGFormer get features from 10X
        # x_512_1 = self.trans_conv(node_image_path_512_fea)
        # if self.use_graph:
        #     x_512_2 = self.graph_conv(node_image_path_512_fea, edge_index_image_512)
        #     if self.aggregate == 'add':
        #         x_512 = self.graph_weight * x_512_2 + (1 - self.graph_weight) * x_512_1
        #     else:
        #         x_512 = torch.cat((x_512_1, x_512_2), dim=1)
        # else:
        #     x_512 = x_512_1
        # x_512 = self.fc(x_512)
        #
        # ###graph_difformer
        # x_512_d = self.difformer(node_image_path_512_fea, edge_index_image_512)
        # x_512_d = self.difformer(x_512, edge_index_image_512)
        x_512_d = self.img_gnn1(node_image_path_512_fea, edge_index_image_512)
        x_512_d = self.gnn_relu(x_512_d)
        #
        # ##SGFormer get features from 20X
        # x_1024_1 = self.trans_conv(node_image_path_1024_fea)
        # if self.use_graph:
        #     x_1024_2 = self.graph_conv(node_image_path_1024_fea, edge_index_image_1024)
        #     if self.aggregate == 'add':
        #         x_1024 = self.graph_weight * x_1024_2 + (1 - self.graph_weight) * x_1024_1
        #     else:
        #         x_1024 = torch.cat((x_1024_1, x_1024_2), dim=1)
        # else:
        #     x_1024 = x_1024_1
        # x_1024 = self.fc(x_1024)

        ###graph_difformer
        # x_1024_d = self.difformer(node_image_path_1024_fea,edge_index_image_1024)
        # x_1024_d = self.difformer(x_1024, edge_index_image_1024)
        x_1024_d = self.img_gnn1(node_image_path_1024_fea, edge_index_image_1024)
        x_1024_d = self.gnn_relu(x_1024_d)

        ###graph_transformer处理TME
        # TME_fea = self.graph_transformer(TME)
        # TME_fea = self.gcn_tme(node_features, edges)
        TME_fea = self.img_gnn2(node_features, edges)
        TME_fea = self.gnn_relu(TME_fea)
        # # ##CoBFormer模型
        # x_100 = self.CoBFormer()

        # x_256_new, x_512_new, x_1024_new,TME_fea_new = x_256_d.unsqueeze(0), x_512_d.unsqueeze(0), x_1024_d.unsqueeze(0), TME_fea.unsqueeze(0)
        x_256_new, x_512_new, x_1024_new, TME_fea_new = x_256_d.transpose(0, 1), x_512_d.transpose(0,
                                                                                                   1), x_1024_d.transpose(
            0, 1), TME_fea.transpose(0, 1)
        # fusion_256 = self.mamba_model1(x_256_new)
        # fusion_512 = self.mamba_model2(x_512_new)
        # fusion_1024 = self.mamba_model3(x_1024_new)
        # TME_fea_a = self.mamba_model1(TME_fea_new)
        x_256_new = self.attention_256(x_256_new)
        x_512_new = self.attention_512(x_512_new)
        x_1024_new = self.attention_1024(x_1024_new)
        TME_fea_new = self.attention_TME(TME_fea_new)

        ####特征整合
        fusion_256 = torch.cat((x_256_new, TME_fea_new), dim=1)
        fusion_512 = torch.cat((x_512_new, TME_fea_new), dim=1)
        fusion_256 = self.atten1(fusion_256, fusion_256)
        fusion_512 = self.atten2(fusion_512, fusion_512)

        ###自适应特征互补融合模块
        adapt = self.adapt_module(fusion_256, fusion_512, x_1024_new, TME_fea_new)

        # 计算每个视图的loss
        out_256 = self.fc1(fusion_256)
        out_512 = self.fc2(fusion_512)
        out_1024 = self.fc3(x_1024_new)
        out_tme = self.fc4(TME_fea_new)
        out_256_hat = torch.argmax(out_256, dim=1)
        out_256_prob = F.softmax(out_256, dim=1)
        out_512_hat = torch.argmax(out_512, dim=1)
        out_512_prob = F.softmax(out_512, dim=1)
        out_1024_hat = torch.argmax(out_1024, dim=1)
        out_1024_prob = F.softmax(out_1024, dim=1)
        out_tme_hat = torch.argmax(out_1024, dim=1)
        out_tme_prob = F.softmax(out_1024, dim=1)
        out_adapt_hat = torch.argmax(adapt, dim=1)
        out_adapt_prob = F.softmax(adapt, dim=1)
        results_dict_256 = {'logits': out_256, 'Y_prob': out_256_prob, 'Y_hat': out_256_hat}
        results_dict_512 = {'logits': out_512, 'Y_prob': out_512_prob, 'Y_hat': out_512_hat}
        results_dict_1024 = {'logits': out_1024, 'Y_prob': out_1024_prob, 'Y_hat': out_1024_hat}
        results_dict_tme = {'logits': out_tme, 'Y_prob': out_tme_prob, 'Y_hat': out_tme_hat}
        results_dict_adapt = {'logits': adapt, 'Y_prob': out_adapt_hat, 'Y_hat': out_adapt_prob}

        return x_256_d, x_512_d, x_1024_d  # results_dict_256, results_dict_512, results_dict_1024, results_dict_tme, results_dict_adapt
        # return results_dict_adapt

    def get_attentions(self, x):
        attns = self.trans_conv.get_attentions(x)  # [layer num, N, N]

        return attns

    def reset_parameters(self):
        self.trans_conv.reset_parameters()
        if self.use_graph:
            self.graph_conv.reset_parameters()


class fusion_model_graph(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 trans_num_layers=1, trans_num_heads=8, trans_dropout=0.2, trans_use_bn=True, trans_use_residual=True,
                 trans_use_weight=True, trans_use_act=True,
                 gnn_num_layers=1, gnn_dropout=0.2, gnn_use_weight=True, gnn_use_init=False, gnn_use_bn=True,
                 gnn_use_residual=True, gnn_use_act=True,
                 use_graph=True, graph_weight=0.8, aggregate='add'):
        super().__init__()

        ##对于TME特征提取
        # self.graph_transformer = GraphTransformer(in_size= 1, num_class=out_channels , d_model=8)
        # self.gcn_tme = GCN_TME(in_channels=11, hidden_channels=2, out_channels=1)
        # ###difformer
        # self.difformer = DIFFormer(in_channels=768, hidden_channels=40, out_channels=1)
        self.img_gnn1 = SAGEConv(in_channels=768, out_channels=1)
        self.img_gnn2 = SAGEConv(in_channels=11, out_channels=1)
        self.gnn_relu = GNN_relu_Block(dim2=1, dropout=0.2)

        # self.attenNN = attenN(hidden_channels, hidden_channels, num_heads=1, kernel='simple',  use_weight=True)
        ##特征融合模块
        # y = model(x)
        self.attention_1024 = self.pooling = nn.AdaptiveMaxPool1d(
            512)  # FixedOutputModel(input_dim=1, hidden_dim=2048, output_dim=128)#TransformerFeatureExtractor(input_dim=1, embed_dim=256, num_heads=8, num_layers=1, chunk_size=256)
        self.attention_512 = self.pooling = nn.AdaptiveMaxPool1d(
            2048)  # 1024,2048 # FixedOutputModel(input_dim=1, hidden_dim=2048, output_dim=256)#TransformerFeatureExtractor(input_dim=1, embed_dim=512, num_heads=8, num_layers=1, chunk_size=256)#TransformerAttention(input_dim=1, embed_dim=800, num_heads=8)
        self.attention_256 = self.pooling = nn.AdaptiveMaxPool1d(
            2048)  # FixedOutputModel(input_dim=1, hidden_dim=2048, output_dim=512)#TransformerFeatureExtractor(input_dim=1, embed_dim=1024, num_heads=8, num_layers=1, chunk_size=256)#TransformerAttention(input_dim=1, embed_dim=1200, num_heads=8)
        self.attention_TME = self.pooling = nn.AdaptiveMaxPool1d(
            1024)  # FixedOutputModel(input_dim=1, hidden_dim=2048, output_dim=1024)

        self.atten = TransformerFeatureExtractor(input_dim=1, embed_dim=1024, num_heads=8, num_layers=1,
                                                 chunk_size=32)  # 1024
        self.atten1 = DIFFormerConv(in_channels=3072, out_channels=1024, num_heads=1, kernel='simple', use_graph=False,
                                    use_weight=True)  # 5120 3072
        self.atten2 = DIFFormerConv(in_channels=3072, out_channels=1024, num_heads=1, kernel='simple', use_graph=False,
                                    use_weight=True)  # 3072   2048

        self.adapt_module = CompleteModel(target_dim=1024, num_classes=out_channels)
        self.fc1 = nn.Linear(1024, out_channels)
        self.fc2 = nn.Linear(1024, out_channels)
        self.fc3 = nn.Linear(512, out_channels)
        self.fc4 = nn.Linear(1024, out_channels)

    def forward(self, node_image_path_256_fea, node_image_path_512_fea, node_image_path_1024_fea, edge_index_image_256,
                edge_index_image_512, edge_index_image_1024, node_features,
                edges):  # node_image_path_256_fea,node_image_path_512_fea,node_image_path_1024_fea,edge_index_image_256,edge_index_image_512,edge_index_image_1024,bag_feats_TME.node,bag_feats_TME.edge_index
        ##SGFormer get features from 20X
        # x_256_1 = self.trans_conv(node_image_path_256_fea)
        # if self.use_graph:
        #     x_256_2 = self.graph_conv(node_image_path_256_fea, edge_index_image_256)
        #     if self.aggregate == 'add':
        #         x_256 = self.graph_weight * x_256_2 + (1 - self.graph_weight) * x_256_1
        #     else:
        #         x_256 = torch.cat((x_256_1, x_256_2), dim=1)
        # else:
        #     x_256 = x_256_1
        # x_256 = self.fc(x_256)
        # ###graph_difformer
        # x_256_d = self.difformer(x_256, edge_index_image_256)
        x_256_d = self.img_gnn1(node_image_path_256_fea, edge_index_image_256)
        x_256_d = self.gnn_relu(x_256_d)

        ##SGFormer get features from 10X
        # x_512_1 = self.trans_conv(node_image_path_512_fea)
        # if self.use_graph:
        #     x_512_2 = self.graph_conv(node_image_path_512_fea, edge_index_image_512)
        #     if self.aggregate == 'add':
        #         x_512 = self.graph_weight * x_512_2 + (1 - self.graph_weight) * x_512_1
        #     else:
        #         x_512 = torch.cat((x_512_1, x_512_2), dim=1)
        # else:
        #     x_512 = x_512_1
        # x_512 = self.fc(x_512)
        #
        # ###graph_difformer
        # x_512_d = self.difformer(node_image_path_512_fea, edge_index_image_512)
        # x_512_d = self.difformer(x_512, edge_index_image_512)
        x_512_d = self.img_gnn1(node_image_path_512_fea, edge_index_image_512)
        x_512_d = self.gnn_relu(x_512_d)
        #
        # ##SGFormer get features from 20X
        # x_1024_1 = self.trans_conv(node_image_path_1024_fea)
        # if self.use_graph:
        #     x_1024_2 = self.graph_conv(node_image_path_1024_fea, edge_index_image_1024)
        #     if self.aggregate == 'add':
        #         x_1024 = self.graph_weight * x_1024_2 + (1 - self.graph_weight) * x_1024_1
        #     else:
        #         x_1024 = torch.cat((x_1024_1, x_1024_2), dim=1)
        # else:
        #     x_1024 = x_1024_1
        # x_1024 = self.fc(x_1024)

        ###graph_difformer
        # x_1024_d = self.difformer(node_image_path_1024_fea,edge_index_image_1024)
        # x_1024_d = self.difformer(x_1024, edge_index_image_1024)
        x_1024_d = self.img_gnn1(node_image_path_1024_fea, edge_index_image_1024)
        x_1024_d = self.gnn_relu(x_1024_d)

        ###graph_transformer处理TME
        # TME_fea = self.graph_transformer(TME)
        # TME_fea = self.gcn_tme(node_features, edges)
        TME_fea = self.img_gnn2(node_features, edges)
        TME_fea = self.gnn_relu(TME_fea)
        # # ##CoBFormer模型
        # x_100 = self.CoBFormer()

        # x_256_new, x_512_new, x_1024_new,TME_fea_new = x_256_d.unsqueeze(0), x_512_d.unsqueeze(0), x_1024_d.unsqueeze(0), TME_fea.unsqueeze(0)
        x_256_new, x_512_new, x_1024_new, TME_fea_new = x_256_d.transpose(0, 1), x_512_d.transpose(0,
                                                                                                   1), x_1024_d.transpose(
            0, 1), TME_fea.transpose(0, 1)
        # fusion_256 = self.mamba_model1(x_256_new)
        # fusion_512 = self.mamba_model2(x_512_new)
        # fusion_1024 = self.mamba_model3(x_1024_new)
        # TME_fea_a = self.mamba_model1(TME_fea_new)
        x_256_new = self.attention_256(x_256_new)
        x_512_new = self.attention_512(x_512_new)
        x_1024_new = self.attention_1024(x_1024_new)
        TME_fea_new = self.attention_TME(TME_fea_new)

        ####特征整合
        fusion_256 = torch.cat((x_256_new, TME_fea_new), dim=1)
        fusion_512 = torch.cat((x_512_new, TME_fea_new), dim=1)
        fusion_256 = self.atten1(fusion_256, fusion_256)
        fusion_512 = self.atten2(fusion_512, fusion_512)

        ###自适应特征互补融合模块
        adapt = self.adapt_module(fusion_256, fusion_512, x_1024_new, TME_fea_new)

        # 计算每个视图的loss
        out_256 = self.fc1(fusion_256)
        out_512 = self.fc2(fusion_512)
        out_1024 = self.fc3(x_1024_new)
        out_tme = self.fc4(TME_fea_new)
        out_256_hat = torch.argmax(out_256, dim=1)
        out_256_prob = F.softmax(out_256, dim=1)
        out_512_hat = torch.argmax(out_512, dim=1)
        out_512_prob = F.softmax(out_512, dim=1)
        out_1024_hat = torch.argmax(out_1024, dim=1)
        out_1024_prob = F.softmax(out_1024, dim=1)
        out_tme_hat = torch.argmax(out_1024, dim=1)
        out_tme_prob = F.softmax(out_1024, dim=1)
        out_adapt_hat = torch.argmax(adapt, dim=1)
        out_adapt_prob = F.softmax(adapt, dim=1)
        results_dict_256 = {'logits': out_256, 'Y_prob': out_256_prob, 'Y_hat': out_256_hat}
        results_dict_512 = {'logits': out_512, 'Y_prob': out_512_prob, 'Y_hat': out_512_hat}
        results_dict_1024 = {'logits': out_1024, 'Y_prob': out_1024_prob, 'Y_hat': out_1024_hat}
        results_dict_tme = {'logits': out_tme, 'Y_prob': out_tme_prob, 'Y_hat': out_tme_hat}
        results_dict_adapt = {'logits': adapt, 'Y_prob': out_adapt_hat, 'Y_hat': out_adapt_prob}

        return results_dict_256, results_dict_512, results_dict_1024, results_dict_tme, results_dict_adapt  # x_256_d, x_512_d, x_1024_d#
        # return results_dict_adapt

    def get_attentions(self, x):
        attns = self.trans_conv.get_attentions(x)  # [layer num, N, N]

        return attns

    def reset_parameters(self):
        self.trans_conv.reset_parameters()
        if self.use_graph:
            self.graph_conv.reset_parameters()


from Models.nystrom_attention import NystromAttention
from Models.TransMIL.net import PPEG, TransLayer


class Intermediate_fusionmodel(nn.Module):
    def __init__(self, input_size_wsi, input_dim_ct, num_classes):
        super().__init__()
        self.pos_layer = PPEG(dim=512)
        self._fc1 = nn.Sequential(nn.Linear(input_size_wsi, 512), nn.ReLU())
        self.cls_token = nn.Parameter(torch.randn(1, 1, 512))
        self.n_classes = num_classes
        self.layer1 = TransLayer(dim=512)
        self.layer2 = TransLayer(dim=512)
        self.norm = nn.LayerNorm(512)
        self._fc2 = nn.Linear(512, self.n_classes)

        self.net_ct = nn.Sequential(
            nn.Linear(input_dim_ct, 512),
            # nn.ReLU(),
            # nn.Linear(64, num_classes)
        )
        self.norm1 = nn.LayerNorm(512)

        self.fc = nn.Sequential(
            nn.Linear(1024, 128), nn.ReLU(), nn.Dropout(0.0),
            nn.Linear(128, num_classes))

    def forward(self, wsi_fea, ct_fea):
        h = wsi_fea.unsqueeze(0)
        # print(h.shape)

        h = self._fc1(h)  # [B, n, 512]

        # ---->pad
        H = h.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        h = torch.cat([h, h[:, :add_length, :]], dim=1)  # [B, N, 512]

        # ---->cls_token
        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1).cuda()
        h = torch.cat((cls_tokens, h), dim=1)

        # ---->Translayer x1
        h = self.layer1(h)  # [B, N, 512]

        # ---->PPEG
        h = self.pos_layer(h, _H, _W)  # [B, N, 512]

        # ---->Translayer x2
        h = self.layer2(h)  # [B, N, 512]

        # ---->cls_token
        h_not_norm = h[:, 0]

        h = self.norm(h)[:, 0]

        fea_ct = self.net_ct(ct_fea)
        fea_ct =self.norm1(fea_ct)

        x = torch.cat([h, fea_ct], dim=1)

        x = self.fc(x)
        return x


class CrossAttention(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.d = dim
        self.w_q = nn.Linear(dim, dim, bias=False)
        self.w_kv = nn.Linear(dim, dim * 2, bias=False)

    def forward(self, x_q, x_kv):
        # x_q, x_kv: [B, D]
        # 转换成序列长度1的形式，并保持 batch-first
        q = self.w_q(x_q).unsqueeze(1)          # [B, 1, D]
        kv = self.w_kv(x_kv).unsqueeze(1)       # [B, 1, 2D]
        k, v = kv.chunk(2, dim=-1)              # [B, 1, D] each
        att = torch.matmul(q, k.transpose(-2,-1)) / math.sqrt(self.d)
        att = F.softmax(att, dim=-1)
        y = torch.matmul(att, v)               # [B, 1, D]
        return y.squeeze(1)


class crossatten_fusionmodel(nn.Module):
    def __init__(self, input_size_wsi, input_dim_ct, num_classes,dim=512):
        super().__init__()
        self.pos_layer = PPEG(dim=512)
        self._fc1 = nn.Sequential(nn.Linear(input_size_wsi, 512), nn.ReLU())
        self.cls_token = nn.Parameter(torch.randn(1, 1, 512))
        self.n_classes = num_classes
        self.layer1 = TransLayer(dim=512)
        self.layer2 = TransLayer(dim=512)
        self.norm = nn.LayerNorm(512)
        self._fc2 = nn.Linear(512, self.n_classes)

        self.net_ct = nn.Sequential(
            nn.Linear(input_dim_ct, 512),
            # nn.ReLU(),
            # nn.Linear(64, num_classes)
        )
        self.norm1 = nn.LayerNorm(512)

        self.cross_ct2wsi = CrossAttention(dim)
        self.cross_wsi2ct = CrossAttention(dim)
        self.mlp = nn.Sequential(
            nn.LayerNorm(dim * 2),
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, num_classes)  # 二分类/回归默认输出1个logit
        )

    def forward(self, wsi_fea, ct_fea):
        h = wsi_fea.unsqueeze(0)
        # print(h.shape)

        h = self._fc1(h)  # [B, n, 512]

        # ---->pad
        H = h.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        h = torch.cat([h, h[:, :add_length, :]], dim=1)  # [B, N, 512]

        # ---->cls_token
        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1).cuda()
        h = torch.cat((cls_tokens, h), dim=1)

        # ---->Translayer x1
        h = self.layer1(h)  # [B, N, 512]

        # ---->PPEG
        h = self.pos_layer(h, _H, _W)  # [B, N, 512]

        # ---->Translayer x2
        h = self.layer2(h)  # [B, N, 512]

        # ---->cls_token
        h_not_norm = h[:, 0]

        h = self.norm(h)[:, 0]

        fea_ct = self.net_ct(ct_fea)
        fea_ct =self.norm1(fea_ct)
        # Cross-Attention 融合
        ct2wsi = self.cross_ct2wsi(h, fea_ct)
        wsi2ct = self.cross_ct2wsi(fea_ct, h)

        # 池化 + 拼接
        fusion = torch.cat([ct2wsi, wsi2ct], dim=1)  # [B, 1024]
        return self.mlp(fusion)

# Bilinear Pooling Fusion
class BPF_fusionmodel(nn.Module):
    def __init__(self, input_size_wsi, input_dim_ct, num_classes,dim=512):
        super().__init__()
        self.pos_layer = PPEG(dim=512)
        self._fc1 = nn.Sequential(nn.Linear(input_size_wsi, 512), nn.ReLU())
        self.cls_token = nn.Parameter(torch.randn(1, 1, 512))
        self.n_classes = num_classes
        self.layer1 = TransLayer(dim=512)
        self.layer2 = TransLayer(dim=512)
        self.norm = nn.LayerNorm(512)
        self._fc2 = nn.Linear(512, self.n_classes)

        self.net_ct = nn.Sequential(
            nn.Linear(input_dim_ct, 512),
            # nn.ReLU(),
            # nn.Linear(64, num_classes)
        )
        self.norm1 = nn.LayerNorm(512)

        self.proj = nn.Linear(dim * dim, num_classes)
        self.mlp = nn.Sequential(
            nn.LayerNorm(dim * 2),
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, num_classes)  # 二分类/回归默认输出1个logit
        )

    def forward(self, wsi_fea, ct_fea):
        h = wsi_fea.unsqueeze(0)
        # print(h.shape)

        h = self._fc1(h)  # [B, n, 512]

        # ---->pad
        H = h.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        h = torch.cat([h, h[:, :add_length, :]], dim=1)  # [B, N, 512]

        # ---->cls_token
        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1).cuda()
        h = torch.cat((cls_tokens, h), dim=1)

        # ---->Translayer x1
        h = self.layer1(h)  # [B, N, 512]

        # ---->PPEG
        h = self.pos_layer(h, _H, _W)  # [B, N, 512]

        # ---->Translayer x2
        h = self.layer2(h)  # [B, N, 512]

        # ---->cls_token
        h_not_norm = h[:, 0]

        h = self.norm(h)[:, 0]

        fea_ct = self.net_ct(ct_fea)
        fea_ct =self.norm1(fea_ct)

        # —— 2. 特征形状此时应为 (batch, 512)
        batch_size = fea_ct.size(0)

        # —— 3. 计算外积: (batch, 512, 1) × (batch, 1, 512) -> (batch, 512, 512)
        ct_exp = fea_ct.unsqueeze(2)  # (batch, 512, 1)
        wsi_exp = h.unsqueeze(1)  # (batch, 1, 512)
        outer = torch.matmul(ct_exp, wsi_exp)  # (batch, 512, 512)

        # —— 4. 展平并投射到 fused_dim
        fused = outer.view(batch_size, -1)  # (batch, 512*512)
        fused = self.proj(fused)  # (batch, fused_dim)

        # —— 5. 归一化（可选）
        # fused = F.normalize(fused, p=2, dim=1)

        return fused

# Late fusion 方法1：特征拼接 + 全连接分类器
class FusionConcat(nn.Module):
    def __init__(self, feature_dim=512, num_classes=2):
        super(FusionConcat, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, feat_ct, feat_wsi):
        fused = torch.cat([feat_ct, feat_wsi], dim=1)
        out = self.classifier(fused)
        return out


# Late fusion 方法1：特征拼接 + 全连接分类器
class FSC_fusionmodel(nn.Module):
    def __init__(self, input_size_wsi, input_dim_ct, num_classes,dim=512):
        super().__init__()
        self.pos_layer = PPEG(dim=512)
        self._fc1 = nn.Sequential(nn.Linear(input_size_wsi, 512), nn.ReLU())
        self.cls_token = nn.Parameter(torch.randn(1, 1, 512))
        self.n_classes = num_classes
        self.layer1 = TransLayer(dim=512)
        self.layer2 = TransLayer(dim=512)
        self.norm = nn.LayerNorm(512)
        self._fc2 = nn.Linear(512, self.n_classes)

        self.net_ct = nn.Sequential(
            nn.Linear(input_dim_ct, 512),
            # nn.ReLU(),
            # nn.Linear(64, num_classes)
        )
        self.norm1 = nn.LayerNorm(512)

        self.classifier = nn.Sequential(
            nn.Linear(dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, wsi_fea, ct_fea):
        h = wsi_fea.unsqueeze(0)
        # print(h.shape)

        h = self._fc1(h)  # [B, n, 512]

        # ---->pad
        H = h.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        h = torch.cat([h, h[:, :add_length, :]], dim=1)  # [B, N, 512]

        # ---->cls_token
        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1).cuda()
        h = torch.cat((cls_tokens, h), dim=1)

        # ---->Translayer x1
        h = self.layer1(h)  # [B, N, 512]

        # ---->PPEG
        h = self.pos_layer(h, _H, _W)  # [B, N, 512]

        # ---->Translayer x2
        h = self.layer2(h)  # [B, N, 512]

        # ---->cls_token
        h_not_norm = h[:, 0]

        h = self.norm(h)[:, 0]

        fea_ct = self.net_ct(ct_fea)
        fea_ct =self.norm1(fea_ct)

        fused = torch.cat([fea_ct, h], dim=1)
        out = self.classifier(fused)

        return out

class FusionWeightedAvg(nn.Module):
    def __init__(self, feature_dim=512, num_classes=2):
        super(FusionWeightedAvg, self).__init__()
        # 学习权重
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, feat_ct, feat_wsi):
        fused = self.alpha * feat_ct + (1 - self.alpha) * feat_wsi
        out = self.classifier(fused)
        return out

# Late fusion 方法2：加权平均特征 + 分类器
class FWA_fusionmodel(nn.Module):
    def __init__(self, input_size_wsi, input_dim_ct, num_classes,dim=512):
        super().__init__()
        self.pos_layer = PPEG(dim=512)
        self._fc1 = nn.Sequential(nn.Linear(input_size_wsi, 512), nn.ReLU())
        self.cls_token = nn.Parameter(torch.randn(1, 1, 512))
        self.n_classes = num_classes
        self.layer1 = TransLayer(dim=512)
        self.layer2 = TransLayer(dim=512)
        self.norm = nn.LayerNorm(512)
        self._fc2 = nn.Linear(512, self.n_classes)

        self.net_ct = nn.Sequential(
            nn.Linear(input_dim_ct, 512),
            # nn.ReLU(),
            # nn.Linear(64, num_classes)
        )
        self.norm1 = nn.LayerNorm(512)

        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.classifier = nn.Linear(dim, num_classes)

    def forward(self, wsi_fea, ct_fea):
        h = wsi_fea.unsqueeze(0)
        # print(h.shape)

        h = self._fc1(h)  # [B, n, 512]

        # ---->pad
        H = h.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        h = torch.cat([h, h[:, :add_length, :]], dim=1)  # [B, N, 512]

        # ---->cls_token
        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1).cuda()
        h = torch.cat((cls_tokens, h), dim=1)

        # ---->Translayer x1
        h = self.layer1(h)  # [B, N, 512]

        # ---->PPEG
        h = self.pos_layer(h, _H, _W)  # [B, N, 512]

        # ---->Translayer x2
        h = self.layer2(h)  # [B, N, 512]

        # ---->cls_token
        h_not_norm = h[:, 0]

        h = self.norm(h)[:, 0]

        fea_ct = self.net_ct(ct_fea)
        fea_ct =self.norm1(fea_ct)

        fused = self.alpha * fea_ct + (1 - self.alpha) * H
        out = self.classifier(fused)

        return out

# Late fusion 方法3：决策层软投票（基于概率）
class FSV_fusionmodel(nn.Module):
    def __init__(self, input_size_wsi, input_dim_ct, num_classes,dim=512):
        super().__init__()
        self.pos_layer = PPEG(dim=512)
        self._fc1 = nn.Sequential(nn.Linear(input_size_wsi, 512), nn.ReLU())
        self.cls_token = nn.Parameter(torch.randn(1, 1, 512))
        self.n_classes = num_classes
        self.layer1 = TransLayer(dim=512)
        self.layer2 = TransLayer(dim=512)
        self.norm = nn.LayerNorm(512)
        self._fc2 = nn.Linear(512, self.n_classes)

        self.net_ct = nn.Sequential(
            nn.Linear(input_dim_ct, 512),
            # nn.ReLU(),
            # nn.Linear(64, num_classes)
        )
        self.norm1 = nn.LayerNorm(512)

        # self.alpha = nn.Parameter(torch.tensor(0.5))
        self.classifier = nn.Linear(dim, num_classes)

    def forward(self, wsi_fea, ct_fea):
        h = wsi_fea.unsqueeze(0)
        # print(h.shape)

        h = self._fc1(h)  # [B, n, 512]

        # ---->pad
        H = h.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        h = torch.cat([h, h[:, :add_length, :]], dim=1)  # [B, N, 512]

        # ---->cls_token
        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1).cuda()
        h = torch.cat((cls_tokens, h), dim=1)

        # ---->Translayer x1
        h = self.layer1(h)  # [B, N, 512]

        # ---->PPEG
        h = self.pos_layer(h, _H, _W)  # [B, N, 512]

        # ---->Translayer x2
        h = self.layer2(h)  # [B, N, 512]

        # ---->cls_token
        h_not_norm = h[:, 0]

        h = self.norm(h)[:, 0]

        fea_ct = self.net_ct(ct_fea)
        fea_ct =self.norm1(fea_ct)

        # 平均概率
        probs_ct = self.classifier(h)
        probs_wsi = self.classifier(fea_ct)
        probs_fused = (probs_ct + probs_wsi) / 2
        preds = torch.argmax(probs_fused, dim=1)
        return probs_fused



import torch
import torch.nn as nn
from .layers.Transformer_EncDec import Encoder, EncoderLayer
from .layers.SelfAttention_Family import FullAttention, AttentionLayer
from .layers.Embed import DataEmbedding_inverted
import numpy as np
from .torchscale.model.LongNet import LongNetEncoder
from .torchscale.architecture.config import EncoderConfig
config1 = EncoderConfig(
    vocab_size=1024,                                         # 输入维度
    segment_length='[2048,4096]',
    dilated_ratio='[1,2]',
    flash_attention=True
)



class FeatureExtractor(nn.Module):
    """
    iTransformer 特征提取器，将 [B, L, N] 输入转换为 [B, N, d_model] 特征表示
    """

    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len       # e.g., 600
        self.n_vars = configs.n_vars         # e.g., 1316
        self.use_norm = configs.use_norm

        self.encoder = LongNetEncoder(config1)


        ######对CT影像数据进行特征提取
        # 输入嵌入，每个 variate token 的 d_model 维表示
        self.enc_embedding = DataEmbedding_inverted(
            configs.seq_len,
            configs.d_model,
            configs.embed,
            configs.freq,
            configs.dropout
        )

        # Encoder-only 架构
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor,
                                      attention_dropout=configs.dropout,
                                      output_attention=False),
                        configs.d_model, configs.n_heads
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for _ in range(configs.e_layers)
            ],
            norm_layer=nn.LayerNorm(configs.d_model)
        )

    def forward(self, x_enc, x_mark_enc=None):

        # 可选归一化（非平稳 Transformer）
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc = x_enc / stdev

        # 嵌入并编码
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B, N, d_model]
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # 去归一化（与训练模型一致）
        if self.use_norm:
            enc_out = enc_out * (stdev[:, 0, :].unsqueeze(1))
            enc_out = enc_out + means[:, 0, :].unsqueeze(1)

        return enc_out, attns














