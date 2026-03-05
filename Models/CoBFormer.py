import torch

# from Model.ffn import *
# from Model.GCN import *
# from Layer.BGA import BGA
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import torch
from torch_geometric.nn import GCNConv, GATConv
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor, matmul
from torch_geometric.utils import degree

import torch
import torch.nn as nn
import torch.nn.functional as F


def sparse_dropout(x: torch.Tensor, p: float, training: bool):
    x = x.coalesce()
    return torch.sparse_coo_tensor(x.indices(), F.dropout(x.values(), p=p, training=training),
                                   size=x.size())


class FFN(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int):
        super(FFN, self).__init__()
        self.lin1 = nn.Linear(in_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, hidden_channels)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # x = self.dropout(x)
        x = F.relu(self.lin1(x))
        return x
        # return self.lin2(x)


class SparseFFN(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int):
        super(SparseFFN, self).__init__()
        self.lin1 = nn.Linear(in_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, hidden_channels)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = sparse_dropout(x, 0.5, self.training)
        x = F.relu(self.lin1(x))
        return self.lin2(x) + x

class GCN(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int,
                 activation, k: int = 2, use_bn=False):
        super(GCN, self).__init__()
        assert k > 1, "k must > 1 !!"
        self.use_bn = use_bn
        self.k = k
        self.conv = nn.ModuleList([GCNConv(in_channels, hidden_channels)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(hidden_channels)])
        for _ in range(1, k - 1):
            self.conv.append(GCNConv(hidden_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
        self.conv.append(GCNConv(hidden_channels, out_channels))
        # self.conv.append(GCNConv(hidden_channels, hidden_channels))
        if activation is None:
            self.activation = F.relu
        else:
            self.activation = activation

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        for i in range(self.k - 1):
            # x = F.dropout(x, p=0.5, training=self.training)
            x = self.conv[i](x, edge_index)
            if self.use_bn:
                x = self.bns[i](x)
            x = self.activation(x)
            x = F.dropout(x, p=0.5, training=self.training)
        return self.conv[-1](x, edge_index)


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
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.5, use_bn=True,
                 use_residual=True, use_weight=True, use_init=False, use_act=True):
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
        self.classifier = nn.Linear(hidden_channels, out_channels)
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
            # layer_.append(x)
        return self.classifier(x)
        # return x


class GAT(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int,
                 activation, n_heads=8, k: int = 2, use_bn=False):
        super(GAT, self).__init__()
        assert k > 1, "k must > 1 !!"
        self.use_bn = use_bn
        self.k = k
        self.conv = nn.ModuleList([GATConv(in_channels, hidden_channels// n_heads, heads=n_heads, dropout=0.6)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(hidden_channels)])
        for _ in range(1, k - 1):
            self.conv.append(GATConv(hidden_channels, hidden_channels // n_heads, heads=n_heads, dropout=0.6))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
        self.conv.append(GATConv(hidden_channels, out_channels, heads=8, concat=False, dropout=0.6))
        # self.conv.append(GCNConv(hidden_channels, hidden_channels))
        self.activation = F.relu

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        for i in range(self.k - 1):
            x = F.dropout(x, p=0.6, training=self.training)
            x = self.conv[i](x, edge_index)
            if self.use_bn:
                x = self.bns[i](x)
            x = self.activation(x)
        x = F.dropout(x, p=0.6, training=self.training)
        return self.conv[-1](x, edge_index)


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        # self.label_same_matrix = torch.load('analysis/label_same_matrix_citeseer.pt').float()

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        # self.label_same_matrix = self.label_same_matrix.to(attn.device)
        # attn = attn * self.label_same_matrix * 2 + attn * (1-self.label_same_matrix)
        attn = self.dropout(F.softmax(attn, dim=-1))
        # attn = self.dropout(attn)

        output = torch.matmul(attn, v)

        return output, attn


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, channels, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        self.n_head = n_head
        self.channels = channels
        d_q = d_k = d_v = channels // n_head

        self.w_qs = nn.Linear(channels, channels, bias=False)
        self.w_ks = nn.Linear(channels, channels, bias=False)
        self.w_vs = nn.Linear(channels, channels, bias=False)
        self.fc = nn.Linear(channels, channels, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        n_head = self.n_head
        d_q = d_k = d_v = self.channels // n_head
        B_q = q.size(0)
        N_q = q.size(1)
        B_k = k.size(0)
        N_k = k.size(1)
        B_v = v.size(0)
        N_v = v.size(1)

        residual = q
        # x = self.dropout(q)

        # Pass through the pre-attention projection: B * N x (h*dv)
        # Separate different heads: B * N x h x dv
        q = self.w_qs(q).view(B_q, N_q, n_head, d_q)
        k = self.w_ks(k).view(B_k, N_k, n_head, d_k)
        v = self.w_vs(v).view(B_v, N_v, n_head, d_v)

        # Transpose for attention dot product: B * h x N x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        # For head axis broadcasting.
        if mask is not None:
            mask = mask.unsqueeze(1)

        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: B x N x h x dv
        # Combine the last two dimensions to concatenate all the heads together: B x N x (h*dv)
        q = q.transpose(1, 2).contiguous().view(B_q, N_q, -1)
        q = self.fc(q)
        q = q + residual

        return q, attn


class FFN(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, channels, dropout=0.1):
        super(FFN, self).__init__()
        self.lin1 = nn.Linear(channels, channels)  # position-wise
        self.lin2 = nn.Linear(channels, channels)  # position-wise
        self.layer_norm = nn.LayerNorm(channels, eps=1e-6)
        self.Dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.layer_norm(x)
        x = self.Dropout(x)
        x = F.relu(self.lin1(x))
        x = self.lin2(x) + residual

        return x


class BGALayer(nn.Module):
    def __init__(self, n_head, channels, use_patch_attn=True, dropout=0.1):
        super(BGALayer, self).__init__()
        self.node_norm = nn.LayerNorm(channels)
        self.node_transformer = MultiHeadAttention(n_head, channels, dropout)
        self.patch_norm = nn.LayerNorm(channels)
        self.patch_transformer = MultiHeadAttention(n_head, channels, dropout)
        self.node_ffn = FFN(channels, dropout)
        self.patch_ffn = FFN(channels, dropout)
        self.fuse_lin = nn.Linear(2 * channels, channels)
        self.use_patch_attn = use_patch_attn
        self.attn = None

    def forward(self, x, patch, attn_mask=None, need_attn=False):
        x = self.node_norm(x)
        patch_x = x[patch]
        patch_x, attn = self.node_transformer(patch_x, patch_x, patch_x, attn_mask)
        patch_x = self.node_ffn(patch_x)
        if need_attn:
            self.attn = torch.zeros((x.shape[0], x.shape[0]))
            for i in tqdm(range(patch.shape[0])):
                p = patch[i].tolist()
                row = torch.tensor([p] * len(p)).T.flatten()
                col = torch.tensor(p * len(p))
                a = attn[i].mean(0).flatten().cpu()
                self.attn = self.attn.index_put((row, col), a)

            self.attn = self.attn[:-1][:, :-1].detach().cpu()

        if self.use_patch_attn:
            p = self.patch_norm(patch_x.mean(dim=1, keepdim=False)).unsqueeze(0)
            p, _ = self.patch_transformer(p, p, p)
            p = self.patch_ffn(p).permute(1, 0, 2)
            #
            p = p.repeat(1, patch.shape[1], 1)
            z = torch.cat([patch_x, p], dim=2)
            patch_x = F.relu(self.fuse_lin(z)) + patch_x

        x[patch] = patch_x

        return x

class BGA(torch.nn.Module):
    def __init__(self, num_nodes: int, in_channels: int, hidden_channels: int, out_channels: int,
                 layers: int, n_head: int, use_patch_attn=True, dropout1=0.5, dropout2=0.1, need_attn=False):
        super(BGA, self).__init__()
        self.layers = layers
        self.n_head = n_head
        self.num_nodes = num_nodes
        self.dropout = nn.Dropout(dropout1)
        self.attribute_encoder = FFN(in_channels, hidden_channels)
        self.BGALayers = nn.ModuleList()
        for _ in range(0, layers):
            self.BGALayers.append(
                BGALayer(n_head, hidden_channels, use_patch_attn, dropout=dropout2))
        self.classifier = nn.Linear(hidden_channels, out_channels)
        self.attn=[]

    def forward(self, x: torch.Tensor, patch: torch.Tensor, need_attn=False):
        patch_mask = (patch != self.num_nodes - 1).float().unsqueeze(-1)
        attn_mask = torch.matmul(patch_mask, patch_mask.transpose(1, 2)).int()

        x = self.attribute_encoder(x)
        for i in range(0, self.layers):
            x = self.BGALayers[i](x, patch, attn_mask, need_attn)
            if need_attn:
                self.attn.append(self.BGALayers[i].attn)
        x = self.dropout(x)
        x = self.classifier(x)
        return x



class CoBFormer(torch.nn.Module):
    def __init__(self, num_nodes: int, in_channels: int, hidden_channels: int, out_channels: int,
                 activation='relu', gcn_layers=2, gcn_type=1, layers=1, n_head=4, dropout1=0.5, dropout2=0.1,
                 alpha=0.8, tau=0.5, gcn_use_bn=False, use_patch_attn=True):
        super(CoBFormer, self).__init__()
        self.alpha = alpha
        self.tau = tau
        self.layers = layers
        self.n_head = n_head
        self.num_nodes = num_nodes
        self.activation = {'relu': F.relu, 'prelu': nn.PReLU()}#({'relu': F.relu, 'prelu': nn.PReLU()})[config['activation']]#activation
        self.dropout = nn.Dropout(dropout1)
        if gcn_type == 1:
            self.gcn = GCN(in_channels, hidden_channels, out_channels, activation, k=gcn_layers, use_bn=gcn_use_bn)
        else:
            self.gcn = GraphConv(in_channels, hidden_channels, out_channels, num_layers=gcn_layers, use_bn=gcn_use_bn)
        # self.gat = GAT(in_channels, hidden_channels, out_channels, activation, k=gcn_layers, use_bn=gcn_use_bn)
        self.bga = BGA(num_nodes, in_channels, hidden_channels, out_channels, layers, n_head,
                                         use_patch_attn, dropout1, dropout2)
        self.attn = None

    def forward(self, x: torch.Tensor, patch: torch.Tensor, edge_index: torch.Tensor, need_attn=False):
        z1 = self.gcn(x, edge_index)
        z2 = self.bga(x, patch, need_attn)
        if need_attn:
            self.attn = self.beyondformer.attn

        return z1, z2

    def loss(self, pred1, pred2, label, mask):
        l1 = F.cross_entropy(pred1[mask], label[mask])
        l2 = F.cross_entropy(pred2[mask], label[mask])
        pred1 *= self.tau
        pred2 *= self.tau
        l3 = F.cross_entropy(pred1[~mask], F.softmax(pred2, dim=1)[~mask])
        l4 = F.cross_entropy(pred2[~mask], F.softmax(pred1, dim=1)[~mask])
        loss = self.alpha * (l1 + l2) + (1 - self.alpha) * (l3 + l4)
        return loss