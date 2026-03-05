# # Copyright (c) 2023 Microsoft
# # Licensed under The MIT License [see LICENSE for details]
#
# from Models.torchscale.architecture.decoder import Decoder, DecoderLayer
# from Models.torchscale.architecture.encoder import Encoder, EncoderLayer
# from Models.torchscale.component.dilated_attention import DilatedAttention
# from fairscale.nn import checkpoint_wrapper, wrap
#
#
# class LongNetDecoderLayer(DecoderLayer):
#
#     def build_self_attention(self, embed_dim, args):
#         return DilatedAttention(
#             args,
#             embed_dim,
#             args.decoder_attention_heads,
#             dropout=args.attention_dropout,
#             self_attention=True,
#             encoder_decoder_attention=False,
#             subln=args.subln,
#         )
#
# class LongNetDecoder(Decoder):
#
#     def build_decoder_layer(
#         self, args, depth, is_moe_layer=False, is_encoder_decoder=False
#     ):
#         layer = LongNetDecoderLayer(
#             args,
#             depth,
#             is_moe_layer=is_moe_layer,
#             is_encoder_decoder=is_encoder_decoder,
#         )
#         if args.checkpoint_activations:
#             layer = checkpoint_wrapper(layer)
#         if args.fsdp:
#             layer = wrap(layer)
#         return layer
#
# class LongNetEncoderLayer(EncoderLayer):
#
#     def build_self_attention(self, embed_dim, args):
#         return DilatedAttention(
#             args,
#             embed_dim,
#             args.encoder_attention_heads,
#             dropout=args.attention_dropout,
#             self_attention=True,
#             encoder_decoder_attention=False,
#             subln=args.subln,
#         )
#
# class LongNetEncoder(Encoder):
#
#     def build_encoder_layer(
#         self, args, depth, is_moe_layer=False, is_encoder_decoder=False
#     ):
#         layer = LongNetEncoderLayer(
#             args,
#             depth,
#             is_moe_layer=is_moe_layer,
#             is_encoder_decoder=is_encoder_decoder,
#         )
#         if args.checkpoint_activations:
#             layer = checkpoint_wrapper(layer)
#         if args.fsdp:
#             layer = wrap(layer)
#         return layer



import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------- 基础组件 ---------
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_mult=4, dropout=0.1):
        super().__init__()
        hidden = int(dim * hidden_mult)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)

def scaled_dot_product_attn(q, k, v, attn_mask=None):
    d = q.shape[-1]
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d)  # [B, h, Lq, Lk]
    if attn_mask is not None:
        scores = scores.masked_fill(~attn_mask, float("-inf"))
    attn = F.softmax(scores, dim=-1)
    out = torch.matmul(attn, v)
    return out

# --------- 膨胀注意力（简化）---------
class DilatedSelfAttention(nn.Module):
    """
    设计要点：
    - 把序列切成若干 segment（长度 seg_len）
    - 段内：dense attention
    - 段间：以 dilation 步长，从相邻若干段中稀疏采样 key/value 参与注意力
    这样做可以模拟 LongNet 的“距离越远、越稀疏”的注意力分配。
    """
    def __init__(self, dim, num_heads=8, seg_len=512, num_neighbors=2, dilation=4, dropout=0.1):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.h = num_heads
        self.dk = dim // num_heads
        self.seg_len = seg_len
        self.num_neighbors = num_neighbors
        self.dilation = dilation

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def _chunk(self, x):
        # x: [B, L, C] -> [B, S, seg, C]，S 为段数
        B, L, C = x.shape
        pad = (self.seg_len - (L % self.seg_len)) % self.seg_len
        if pad > 0:
            x = F.pad(x, (0, 0, 0, pad))  # pad on length dim
        S = x.shape[1] // self.seg_len
        x = x.view(B, S, self.seg_len, C)
        return x, pad, S

    def forward(self, x):
        """
        x: [B, L, C]
        return: [B, L, C]
        """
        B, L, C = x.shape

        # 线性映射 + multi-head 视角
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q, pad, S = self._chunk(q)
        k, _, _ = self._chunk(k)
        v, _, _ = self._chunk(v)   # 全部切段: [B, S, seg, C]

        # 变成多头
        def to_heads(t):
            # [B, S, seg, C] -> [B, S, h, seg, dk]
            B, S, seg, C = t.shape
            t = t.view(B, S, seg, self.h, self.dk).permute(0, 1, 3, 2, 4)
            return t.contiguous()
        q = to_heads(q)
        k = to_heads(k)
        v = to_heads(v)

        # 段内 dense + 段间 dilated 稀疏
        # 结果容器
        out = torch.zeros_like(q)  # [B, S, h, seg, dk]

        # 先做段内 dense
        # q_s, k_s, v_s: [B, h, seg, dk]
        for s in range(S):
            q_s = q[:, s]  # [B, h, seg, dk]
            k_s = k[:, s]
            v_s = v[:, s]
            # 段内注意力
            local = scaled_dot_product_attn(q_s, k_s, v_s)
            out[:, s] = local

        # 再做段间 dilated（与相邻 num_neighbors 段做稀疏注意力）
        # 我们将“被采样的 key/value 索引”设为每隔 self.dilation 个 token 取一个
        dil_idx = torch.arange(0, self.seg_len, self.dilation, device=x.device)  # [seg//d]

        for s in range(S):
            for offset in range(1, self.num_neighbors + 1):
                for neigh in (s - offset, s + offset):
                    if 0 <= neigh < S:
                        q_s = q[:, s]                               # [B, h, seg, dk]
                        k_n = k[:, neigh, :, dil_idx, :]            # [B, h, seg//d, dk]
                        v_n = v[:, neigh, :, dil_idx, :]
                        # 构造 mask（全可见，这里不做额外掩码）
                        # 做一次 cross-segment 稀疏注意力并累加
                        cross = scaled_dot_product_attn(q_s, k_n, v_n)
                        out[:, s] = out[:, s] + self.drop(cross) * (1.0 / (2 * self.num_neighbors))

        # 合并 heads
        # [B, S, h, seg, dk] -> [B, S, seg, C]
        out = out.permute(0, 1, 3, 2, 4).contiguous().view(B, S, self.seg_len, C)

        # 拼回原长度
        out = out.view(B, S * self.seg_len, C)
        if pad > 0:
            out = out[:, :L, :]
        return self.o_proj(out)  # 线性输出


class LongNetBlock(nn.Module):
    def __init__(self, dim, heads=8, seg_len=512, neighbors=2, dilation=4, ff_mult=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = DilatedSelfAttention(dim, heads, seg_len, neighbors, dilation, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, ff_mult, dropout)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


class SimpleLongNetEncoder(nn.Module):
    """
    一个可处理超长序列的 LongNet 风格 Encoder：
    - 位置编码用可学习偏置（可按需替换成相对位置）
    - N 层 Dilated Self-Attention，层间可加深 dilation 或 neighbors
    """
    def __init__(self, dim=1536, depth=6, heads=8,
                 seg_len=512, neighbors=2, base_dilation=4, ff_mult=4, dropout=0.1):
        super().__init__()
        self.pos_bias = nn.Parameter(torch.zeros(1, 1, dim))  # 简化版位置偏置
        blocks = []
        for i in range(depth):
            # 逐层扩大 dilation（可选）
            dilation = base_dilation * (2 ** i)
            blocks.append(
                LongNetBlock(dim, heads, seg_len, neighbors, dilation, ff_mult, dropout)
            )
        self.blocks = nn.ModuleList(blocks)
        self.norm = nn.LayerNorm(dim)
        self.fc = nn.Linear(dim, 1)

    def forward(self, x):
        """
        x: [B, L, 1536]  —— 你的输入是 [20000, 1536]，请加 batch 维度
        """
        x = x + self.pos_bias   # 简化版位置建模
        for blk in self.blocks:
            x = blk(x)
        a = self.norm(x)     # [B, L, 1536]
        a = self.fc(a).squeeze(-1)
        pooled = torch.mean(x, dim=1)  # 对L维度求均值，输出 [B, C]
        return pooled.squeeze(0), a.squeeze(0)
