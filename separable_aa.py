"""
ELViT v2
"""
import copy
import itertools
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import DropPath, trunc_normal_

import prompters

width = [32, 64, 144, 288]
depth = [4, 4, 12, 8]
vit_num = [0, 0, 4, 4]
# 12m
mlp_ratios = {
    '0': [4, 4, 4, 4],
    '1': [4, 4, 4, 4],
    '2': [4, 4, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4],
    '3': [4, 4, 3, 3, 3, 3, 4, 4],
}

input_size = 224


def get_mask(input_resolution, window_size):
    attn_map = F.unfold(torch.ones([1, 1, input_resolution[0], input_resolution[1]]), window_size,
                        dilation=1, padding=(window_size // 2, window_size // 2),
                        stride=1)  # [1,9,196] / [1,9,49]
    attn_mask = (attn_map.squeeze(0).permute(1, 0)) == 0  # [196,9] / [49,9]
    return attn_mask


class LinearAggregatedAttention(nn.Module):
    def __init__(self, dim, input_resolution, num_heads=8, window_size=3, qkv_bias=True,
                 attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = dim ** -0.5
        self.sr_ratio = sr_ratio

        self.window_size = window_size
        self.local_len = window_size ** 2
        self.H, self.W = input_resolution[0], input_resolution[1]
        self.pool_H, self.pool_W = input_resolution[0] // self.sr_ratio, input_resolution[1] // self.sr_ratio
        self.pool_len = self.pool_H * self.pool_W

        self.unfold = nn.Unfold(kernel_size=window_size, padding=window_size // 2, stride=1)

        self.q = nn.Linear(dim, num_heads, bias=qkv_bias)
        self.query_embedding = nn.Parameter(
            nn.init.trunc_normal_(torch.empty(self.num_heads, 1, 1), mean=0, std=0.02))  # [8,1,1]
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        # self.talking_head = nn.Conv2d(self.num_heads, self.num_heads, kernel_size=1, stride=1, padding=0)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Components to generate pooled features.
        self.pool = nn.AdaptiveAvgPool2d((self.pool_H, self.pool_W))
        self.sr = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0)
        self.norm = nn.LayerNorm(dim)
        self.act = nn.GELU()

        # mlp to generate continuous relative position bias -> learnable bias for global features
        self.learnable_pos_bias_pool = nn.Parameter(nn.init.trunc_normal_(torch.empty(num_heads, self.H * self.W,
                                                                                      self.pool_len), mean=0,
                                                                          std=0.0004))
        # relative bias for local features
        self.relative_pos_bias_local = nn.Parameter(
            nn.init.trunc_normal_(torch.empty(num_heads, self.local_len), mean=0, std=0.0004))

        # Generate padding_mask && sequnce length scale
        padding_mask = get_mask(input_resolution, window_size)
        self.register_buffer("padding_mask", padding_mask, persistent=False)

        # dynamic_local_bias:
        self.learnable_tokens = nn.Parameter(
            nn.init.trunc_normal_(torch.empty(num_heads, 1, self.local_len), mean=0,
                                  std=0.02))  # [4,1,9] / [8,1,9]
        self.learnable_bias = nn.Parameter(torch.zeros(num_heads, 1, self.local_len))

    def forward(self, x):
        B, N, C = x.shape  # [B,196,144] / [B,49,288]

        # Generate queries,  and add query embedding
        q_local = self.q(x).reshape(B, N, self.num_heads, 1).permute(0, 2, 1, 3) + self.query_embedding
        # self.q->Linear(C,C)
        # 3 [B,196,144]->[B,4,196,1]
        # 4 [B,49,288]->[B,8,49,1]

        # Generate unfolded keys and values and l2-normalize them 此处就是文中提到的滑动窗的key和value生成部分，由其生成滑动窗的局部注意力
        kv_local = self.kv(x).permute(0, 2, 1).reshape(B, -1, self.H, self.W)
        # [B,288,14,14] / [B,576,7,7]
        # unfold作用就是将数据按滑动窗大小(3*3)重新进行排列
        k_local, v_local = self.unfold(kv_local).reshape(
            B, 2 * self.num_heads, self.head_dim, self.local_len, N).permute(0, 1, 4, 2, 3).chunk(2, dim=1)
        # [B,4,196,36,9] [B,4,196,36,9] / [B,8,49,36,9] [B,8,49,36,9]

        # Compute local similarity
        context_score = F.softmax(q_local, dim=2)
        context_vector = torch.sum(k_local, dim=-2) * context_score * self.scale
        attn_local = (context_vector + self.relative_pos_bias_local.unsqueeze(1)).masked_fill(self.padding_mask,
                                                                                              float('-inf'))
        # [B,4,196,9] / [B,8,49,9]

        # Generate pooled features 此处就是文中提到的激活和池化，先卷积->激活->池化
        x_ = x.permute(0, 2, 1).reshape(B, -1, self.H,
                                        self.W).contiguous()  # [B,144,14,14] / [B,288,7,7]
        x_ = self.pool(self.act(self.sr(x_))).reshape(B, -1, self.pool_len).permute(0, 2, 1)
        # 3 self.sr->Conv2d(144, 144, kernel_size=(1, 1), stride=(1, 1)) [B,144,14,14]->[B,49,144]
        # 4 self.sr->Conv2d(288, 288, kernel_size=(1, 1), stride=(1, 1)) [B,288,14,14]->[B,49,288]
        x_ = self.norm(x_)

        # Generate pooled keys and values
        kv_pool = self.kv(x_).reshape(B, self.pool_len, 2 * self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        # [B,49,144]->[B,8,49,36] / [B,49,288]->[B,16,49,36]
        k_pool, v_pool = kv_pool.chunk(2, dim=1)
        # [B,4,49,36] [B,4,49,36] / [B,8,49,36] [B,8,49,36]

        # Compute pooled similarity
        k_pool = torch.sum(k_pool, dim=-1, keepdim=True)
        attn_pool = (context_score @ k_pool.transpose(-2, -1) * self.scale + self.learnable_pos_bias_pool)
        # [B,4,196,49] /[B,8,49,49]

        # Concatenate local & pooled similarity matrices and calculate attention weights through the same Softmax
        attn = torch.cat([attn_local, attn_pool], dim=-1).softmax(dim=-1)
        # [B,4,196,58] / [B,8,49,58]
        attn = self.attn_drop(attn)

        # Split the attention weights and separately aggregate the values of local & pooled features
        attn_local, attn_pool = torch.split(attn, [self.local_len, self.pool_len], dim=-1)
        # [B,4,196,9] [B,4,196,49] / [B,8,49,9] [B,8,49,49]
        x_local = torch.sum((self.learnable_tokens * context_score) + attn_local + self.learnable_bias, dim=2,
                            keepdim=True)
        x_local = torch.sum(F.relu(v_local) * x_local.unsqueeze(-2), dim=-1)
        # [B,4,196,36] / [B,8,49,36]
        x_pool = attn_pool @ v_pool  # [B,4,196,36] / [B,8,49,36]
        x = (x_local + x_pool).transpose(1, 2).reshape(B, N, C)  # [B,196,144] / [B,49,288]

        # Linear projection and output
        x = self.proj(x)
        # 3 Linear(in_features=144, out_features=144, bias=True)
        # 4 Linear(in_features=288, out_features=288, bias=True)
        x = self.proj_drop(x)

        return x


class AggregatedAttention(nn.Module):

    def __init__(self, dim, input_resolution, num_heads=8, window_size=3, qkv_bias=True,
                 attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = dim ** -0.5
        self.sr_ratio = sr_ratio

        self.window_size = window_size
        self.local_len = window_size ** 2
        self.H, self.W = input_resolution[0], input_resolution[1]
        self.pool_H, self.pool_W = input_resolution[0] // self.sr_ratio, input_resolution[1] // self.sr_ratio
        self.pool_len = self.pool_H * self.pool_W

        self.unfold = nn.Unfold(kernel_size=window_size, padding=window_size // 2, stride=1)

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.query_embedding = nn.Parameter(
            nn.init.trunc_normal_(torch.empty(self.num_heads, 1, self.head_dim), mean=0, std=0.02))  # [3,1,24]
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Components to generate pooled features.
        self.pool = nn.AdaptiveAvgPool2d((self.pool_H, self.pool_W))
        self.sr = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0)
        self.norm = nn.LayerNorm(dim)
        self.act = nn.GELU()

        # mlp to generate continuous relative position bias -> learnable bias for global features
        self.learnable_pos_bias_pool = nn.Parameter(nn.init.trunc_normal_(torch.empty(num_heads, self.pool_len),
                                                                          mean=0, std=0.0004))
        # relative bias for local features
        self.relative_pos_bias_local = nn.Parameter(
            nn.init.trunc_normal_(torch.empty(num_heads, self.local_len), mean=0,
                                  std=0.0004))

        # Generate padding_mask && sequnce length scale
        padding_mask = get_mask(input_resolution, window_size)
        self.register_buffer("padding_mask", padding_mask, persistent=False)

        # dynamic_local_bias:
        self.learnable_tokens = nn.Parameter(
            nn.init.trunc_normal_(torch.empty(num_heads, self.head_dim, self.local_len), mean=0,
                                  std=0.02))  # [3,24,9] / [6,24,9] / [12,24,9]
        self.learnable_bias = nn.Parameter(torch.zeros(num_heads, 1, self.local_len))

    def forward(self, x):
        B, N, C = x.shape  # [B,3136,72] / [B,784,144] / [B,196,288]

        # Generate queries,  and add query embedding
        q_local = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3) + self.query_embedding
        # self.q->Linear(C,C)
        # 1 [B,3136,72]->[B,3,3136,24]
        # 2 [B,784,144]->[B,6,784,24]
        # 3 [B,196,288]->[B,12,196,24]

        # Generate unfolded keys and values and l2-normalize them 此处就是文中提到的滑动窗的key和value生成部分，由其生成滑动窗的局部注意力
        kv_local = self.kv(x).permute(0, 2, 1).reshape(B, -1, self.H, self.W)
        # [B,144,56,56] / [B,288,28,28] / [B,576,14,14]
        # unfold作用就是将数据按滑动窗大小(3*3)重新进行排列
        k_local, v_local = self.unfold(kv_local).reshape(
            B, 2 * self.num_heads, self.head_dim, self.local_len, N).permute(0, 1, 4, 2, 3).chunk(2, dim=1)
        # [B,3,3136,24,9] [B,3,3136,24,9] / [B,6,784,24,9] [B,6,784,24,9] / [B,12,196,24,9] [B,12,196,24,9]

        # Compute local similarity
        attn_local = ((q_local.unsqueeze(-2) @ k_local * self.scale).squeeze(-2)
                      + self.relative_pos_bias_local.unsqueeze(1)).masked_fill(self.padding_mask, float('-inf'))
        # [B,3,3136,9] / [B,6,784,9] / [B,12,196,9]

        # Generate pooled features 此处就是文中提到的激活和池化，先卷积->激活->池化
        x_ = x.permute(0, 2, 1).reshape(B, -1, self.H,
                                        self.W).contiguous()  # [B,72,56,56] / [B,144,28,28] / [B,288,14,14]
        x_ = self.pool(self.act(self.sr(x_))).reshape(B, -1, self.pool_len).permute(0, 2, 1)
        # 1 self.sr->Conv2d(72, 72, kernel_size=(1, 1), stride=(1, 1)) [B,72,56,56]->[B,49,72]
        # 2 self.sr->Conv2d(144, 144, kernel_size=(1, 1), stride=(1, 1)) [B,144,28,28]->[B,49,144]
        # 3 self.sr->Conv2d(288, 288, kernel_size=(1, 1), stride=(1, 1)) [B,288,14,14]->[B,49,288]
        x_ = self.norm(x_)

        # Generate pooled keys and values
        kv_pool = self.kv(x_).reshape(B, self.pool_len, 2 * self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        # [B,49,72]->[B,6,49,24] / [B,49,144]->[B,12,49,24] / [B,49,288]->[B,24,49,24]
        k_pool, v_pool = kv_pool.chunk(2, dim=1)
        # [B,3,49,24] [B,3,49,24] / [B,6,49,24] [B,6,49,24] / [B,12,49,24] [B,12,49,24]

        # Compute pooled similarity
        attn_pool = self.scale * q_local @ k_pool.transpose(-2, -1) + self.learnable_pos_bias_pool.unsqueeze(1)
        # [B,3,3136,49] / [B,6,784,49] /[B,12,196,49]

        # Concatenate local & pooled similarity matrices and calculate attention weights through the same Softmax
        attn = torch.cat([attn_local, attn_pool], dim=-1).softmax(dim=-1)
        # [B,3,3136,58] / [B,6,784,58] / [B,12,196,58]
        attn = self.attn_drop(attn)

        # Split the attention weights and separately aggregate the values of local & pooled features
        attn_local, attn_pool = torch.split(attn, [self.local_len, self.pool_len], dim=-1)
        # [B,3,3136,9] [B,3,3136,49] / [B,6,784,9] [B,6,784,49] / [B,12,196,9] [B,12,196,49]
        x_local = (((q_local @ self.learnable_tokens) + self.learnable_bias + attn_local).unsqueeze(-2)
                   @ v_local.transpose(-2, -1)).squeeze(-2)
        # [B,3,3136,24] / [B,6,784,24] / [B,12,196,24]
        x_pool = attn_pool @ v_pool  # [B,3,3136,24] / [B,6,784,24] / [B,12,196,24]
        x = (x_local + x_pool).transpose(1, 2).reshape(B, N, C)  # [B,3136,72] / [B,784,144] / [B,196,288]

        # Linear projection and output
        x = self.proj(x)
        # 1 Linear(in_features=72, out_features=72, bias=True)
        # 2 Linear(in_features=144, out_features=144, bias=True)
        # 3 Linear(in_features=288, out_features=288, bias=True)
        x = self.proj_drop(x)

        return x


class BasicAttention(torch.nn.Module):
    def __init__(self, dim=384, key_dim=32, num_heads=8,
                 attn_ratio=4,
                 resolution=7):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio
        h = self.dh + nh_kd * 2
        self.N = resolution ** 2
        self.N2 = self.N
        self.qkv = nn.Linear(dim, h)
        self.proj = nn.Linear(self.dh, dim)

        points = list(itertools.product(range(resolution), range(resolution)))
        N = len(points)
        attention_offsets = {}
        idxs = []
        for p1 in points:
            for p2 in points:
                offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        self.attention_biases = torch.nn.Parameter(
            torch.zeros(num_heads, len(attention_offsets)))
        self.register_buffer('attention_bias_idxs',
                             torch.LongTensor(idxs).view(N, N))

    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        if mode and hasattr(self, 'ab'):
            del self.ab
        else:
            self.ab = self.attention_biases[:, self.attention_bias_idxs]

    def forward(self, x):  # x (B,N,C)
        B, N, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.reshape(B, N, self.num_heads, -1).split([self.key_dim, self.key_dim, self.d], dim=3)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        attn = (
                (q @ k.transpose(-2, -1)) * self.scale
                +
                (self.attention_biases[:, self.attention_bias_idxs]
                 if self.training else self.ab)
        )
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, self.dh)
        x = self.proj(x)
        return x


class SeparableAttn(torch.nn.Module):

    def __init__(self, dim=384, key_dim=None, attn_dropout=0., heads=8, bias: bool = True, token_len=49):
        super().__init__()
        self.nh = heads
        if key_dim is None:
            key_dim = int(dim / heads)
        self.qkv_proj = nn.Linear(
            in_features=dim,
            out_features=(1 + key_dim * 2) * self.nh,
            bias=bias
        )

        self.attn_dropout = nn.Dropout(p=attn_dropout)
        self.cv_proj = nn.Linear(
            in_features=token_len,
            out_features=1,
            bias=False
        )
        self.out_proj = nn.Linear(
            in_features=key_dim * self.nh,
            out_features=dim,
            bias=bias
        )
        self.embed_dim = key_dim

    def forward(self, x):  # (B,N,C)
        B, N, C = x.shape
        # (B,N,C)->(B,N,(1+2*kd)*nh)
        mask = [0, 0, 0, 0, 0, 0, 0, 0]
        qkv = self.qkv_proj(x)
        # Query [B,N,nh,1]
        # key [B,N,nh,kd]
        # value [B,N,nh,kd]
        query, key, value = torch.split(qkv.view(B, N, self.nh, -1),
                                        split_size_or_sections=[1, self.embed_dim, self.embed_dim], dim=-1)
        query = query.permute(0, 2, 1, 3)  # [b,nh,n,d]
        key = key.permute(0, 2, 1, 3)
        value = value.permute(0, 2, 1, 3)
        # apply softmax along N dimension
        context_scores = F.softmax(query, dim=2)
        context_scores = self.attn_dropout(context_scores)

        # Compute context vector
        # [B, nh, N, kd] x [B, nh, N, 1] -> [B, nh, N, kd]
        context_vector = key * context_scores
        # [B, nh, N, kd] --> [B, nh, 1, kd]
        # context_vector = torch.sum(context_vector, dim=2, keepdim=True)
        context_vector = self.cv_proj(context_vector.transpose(2, 3)).transpose(2, 3)

        # combine context vector with values
        # [B, nh, N, kd] * [B, nh, 1, kd] --> [B, nh, N, kd]
        out = F.relu(value) * context_vector.expand_as(value)
        out = out.permute(0, 2, 1, 3)  # [B, N, nh*kd]
        for i, m in enumerate(mask):
            if m == 1:
                out[:, :, i, :] = torch.zeros(B, N, self.embed_dim)
        out = out.reshape(B, N, -1)
        out = self.out_proj(out)
        return out


def stem(in_chs, out_chs):
    return nn.Sequential(
        nn.Conv2d(in_chs, out_chs, kernel_size=7, stride=4, padding=3),
        nn.BatchNorm2d(out_chs),
        nn.ReLU(), )


class Embedding(nn.Module):
    """
    Patch Embedding that is implemented by a layer of conv.
    Input: tensor in shape [B, C, H, W]
    Output: tensor in shape [B, C, H/stride, W/stride]
    """

    def __init__(self, patch_size=3, stride=2, padding=1,
                 in_chans=3, embed_dim=768, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=(patch_size, patch_size),
                              stride=(stride, stride), padding=(padding, padding))
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x


class Flat(nn.Module):

    def __init__(self, ):
        super().__init__()

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        return x


class UnFlat(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        bs, num_token, dim = x.shape
        x = x.transpose(1, 2).view(bs, dim, int(math.sqrt(num_token)), -1)
        return x


class GLU(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = int(2 * hidden_features / 3)
        self.fc1 = nn.Linear(in_features, hidden_features * 2)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x, v = self.fc1(x).chunk(2, dim=-1)
        x = self.act(x) * v
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Mlp(nn.Module):
    """
    Implementation of MLP with 1*1 convolution.
    Input: tensor with shape [B, C, H, W]
    """

    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU, drop=0., mid_conv=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.mid_conv = mid_conv
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

        if self.mid_conv:
            self.mid = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1,
                                 groups=hidden_features)
            self.mid_norm = nn.BatchNorm2d(hidden_features)

        self.norm1 = nn.BatchNorm2d(hidden_features)
        self.norm2 = nn.BatchNorm2d(out_features)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.norm1(x)
        x = self.act(x)

        if self.mid_conv:
            x_mid = self.mid(x)
            x_mid = self.mid_norm(x_mid)
            x = self.act(x_mid)
        x = self.drop(x)

        x = self.fc2(x)
        x = self.norm2(x)

        x = self.drop(x)
        return x


class Meta3D(nn.Module):

    def __init__(self, dim, input_resolution, mlp_ratio=4., num_heads=8, sr_ratio=1,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 drop=0., drop_path=0.,
                 use_layer_scale=True, layer_scale_init_value=1e-5):

        super().__init__()

        self.norm1 = norm_layer(dim)
        self.token_mixer = LinearAggregatedAttention(dim=dim, input_resolution=input_resolution, num_heads=num_heads,
                                                     sr_ratio=sr_ratio)
        # self.token_mixer = AggregatedAttention(dim=dim, input_resolution=input_resolution, num_heads=num_heads,
        #                                              sr_ratio=sr_ratio)
        # self.token_mixer = BasicAttention(dim=dim, num_heads=num_heads, resolution=input_resolution[0])
        # self.token_mixer = SeparableAttn(dim=dim, attn_dropout=drop, heads=num_heads, token_len=input_resolution[0] ** 2)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = GLU(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)
        # self.mlp = LinearMlp(in_features=dim, hidden_features=mlp_hidden_dim,
        #                      act_layer=act_layer, drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. \
            else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(
                layer_scale_init_value * torch.ones(dim), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(
                layer_scale_init_value * torch.ones(dim), requires_grad=True)

    def forward(self, x):
        if self.use_layer_scale:
            x = x + self.drop_path(
                self.layer_scale_1.unsqueeze(0).unsqueeze(0)
                * self.token_mixer(self.norm1(x)))
            x = x + self.drop_path(
                self.layer_scale_2.unsqueeze(0).unsqueeze(0)
                * self.mlp(self.norm2(x)))

        else:
            x = x + self.drop_path(self.token_mixer(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Meta4D(nn.Module):

    def __init__(self, dim, mlp_ratio=4.,
                 act_layer=nn.GELU,
                 drop=0., drop_path=0.,
                 use_layer_scale=True, layer_scale_init_value=1e-5):
        super().__init__()

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop, mid_conv=True)

        self.drop_path = DropPath(drop_path) if drop_path > 0. \
            else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_2 = nn.Parameter(
                layer_scale_init_value * torch.ones(dim), requires_grad=True)

    def forward(self, x):
        if self.use_layer_scale:
            x = x + self.drop_path(
                self.layer_scale_2.unsqueeze(-1).unsqueeze(-1)
                * self.mlp(x))
        else:
            x = x + self.drop_path(self.mlp(x))
        return x


def meta_blocks(dim, index, layers, input_resolution, extra_block=None, meta_len=0,
                mlp_ratios=None, num_heads=8, sr_ratio=1,
                act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                drop_rate=.0, drop_path_rate=0.,
                use_layer_scale=True, layer_scale_init_value=1e-5, vit_num=[0, 0, 0, 0]):
    blocks = []
    if vit_num[index] == layers[index]:
        blocks.append(Flat())
    for block_idx in range(layers[index]):
        block_dpr = drop_path_rate * (
                block_idx + sum(layers[:index])) / (sum(layers) - 1)
        mlp_ratio = mlp_ratios[str(index)][block_idx]
        if layers[index] - block_idx <= vit_num[index]:
            blocks.append(extra_block(input_resolution[0], meta_len, dim) if extra_block else nn.Identity())
            blocks.append(Meta3D(
                dim, mlp_ratio=mlp_ratio, input_resolution=input_resolution, num_heads=num_heads, sr_ratio=sr_ratio,
                act_layer=act_layer, norm_layer=norm_layer,
                drop=drop_rate, drop_path=block_dpr,
                use_layer_scale=use_layer_scale,
                layer_scale_init_value=layer_scale_init_value,
            ))
        else:
            blocks.append(Meta4D(
                dim, mlp_ratio=mlp_ratio,
                act_layer=act_layer,
                drop=drop_rate, drop_path=block_dpr,
                use_layer_scale=use_layer_scale,
                layer_scale_init_value=layer_scale_init_value,
            ))
            if vit_num[index] and layers[index] - block_idx - 1 == vit_num[index]:
                blocks.append(Flat())
    if vit_num[index] and index < len(vit_num) - 1:
        blocks.append(UnFlat())
    blocks = nn.Sequential(*blocks)
    return blocks


class Model(nn.Module):

    def __init__(self, layers, embed_dims=None, extra_method=None, meta_len=0,
                 mlp_ratios=None, downsamples=None,
                 norm_layer=nn.LayerNorm, act_layer=nn.GELU,
                 num_classes=1000,
                 down_patch_size=3, down_stride=2, down_pad=1,
                 drop_rate=0., drop_path_rate=[0., 0., 0., 0.],
                 use_layer_scale=True, layer_scale_init_value=1e-5,
                 init_cfg=None,
                 pretrained=None,
                 vit_num=[0, 0, 0, 0], sr_ratio=[8, 4, 2, 1], num_heads=[1, 2, 4, 8],
                 distillation=False,
                 **kwargs):
        super().__init__()
        self.num_classes = num_classes

        self.patch_embed = stem(3, embed_dims[0])
        self.extra_block = prompters.__dict__[extra_method] if extra_method else None
        network = []
        for i in range(len(layers)):
            input_resolution = input_size // (2 ** (i + 2))
            stage = meta_blocks(embed_dims[i], i, layers, extra_block=self.extra_block, meta_len=meta_len,
                                mlp_ratios=mlp_ratios,
                                act_layer=act_layer, norm_layer=norm_layer,
                                drop_rate=drop_rate,
                                drop_path_rate=drop_path_rate[i],
                                use_layer_scale=use_layer_scale,
                                layer_scale_init_value=layer_scale_init_value,
                                vit_num=vit_num,
                                input_resolution=(input_resolution, input_resolution),
                                sr_ratio=sr_ratio[i],
                                num_heads=num_heads[i])
            network.append(stage)
            if i >= len(layers) - 1:
                break
            if downsamples[i] or embed_dims[i] != embed_dims[i + 1]:
                # downsampling between two stages
                network.append(
                    Embedding(
                        patch_size=down_patch_size, stride=down_stride,
                        padding=down_pad,
                        in_chans=embed_dims[i], embed_dim=embed_dims[i + 1]
                    )
                )

        self.network = nn.ModuleList(network)

        # Classifier head
        self.norm = norm_layer(embed_dims[-1])
        self.head = nn.Linear(
            embed_dims[-1], num_classes) if num_classes > 0 \
            else nn.Identity()
        self.dist = distillation
        if self.dist:
            self.dist_head = nn.Linear(
                embed_dims[-1], num_classes) if num_classes > 0 \
                else nn.Identity()

        self.apply(self.cls_init_weights)

        self.init_cfg = copy.deepcopy(init_cfg)
        # load pre-trained model
        # if pretrained is not None:
        #     self.init_weights()

    # init for classification
    def cls_init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward_tokens(self, x, meta=None):
        for idx, block in enumerate(self.network):
            if isinstance(block, nn.Sequential) and self.extra_block:
                for b in block:
                    if isinstance(b, self.extra_block):
                        x = b(x, meta)
                    else:
                        x = b(x)
            else:
                x = block(x)
        return x

    def forward(self, x, meta=None):
        x = self.patch_embed(x)
        x = self.forward_tokens(x, meta)
        x = self.norm(x)
        if self.dist:
            cls_out = self.head(x.mean(-2)), self.dist_head(x.mean(-2))
            if not self.training:
                cls_out = (cls_out[0] + cls_out[1]) / 2
        else:
            cls_out = self.head(x.mean(-2))
        # for image classification
        return cls_out


# @register_model
def lightweight_model(extra_method=None, meta_len=0, pretrained=False, num_classes=1000, distillation=False, **kwargs):
    model = Model(
        extra_method=extra_method,
        meta_len=meta_len,
        layers=depth,
        embed_dims=width,
        downsamples=[True, True, True, True],
        vit_num=vit_num,
        num_classes=num_classes,
        pretrained=pretrained,
        drop_path_rate=[0.1, 0.1, 0.1, 0.1],
        distillation=distillation,
        mlp_ratios=mlp_ratios,
        **kwargs)
    return model
