from math import sqrt

import torch
import torch.nn as nn


class ResNormLayer(nn.Module):
    def __init__(self, linear_size, ):
        super(ResNormLayer, self).__init__()
        self.l_size = linear_size
        self.nonlin1 = nn.ReLU(inplace=True)
        self.nonlin2 = nn.ReLU(inplace=True)
        self.norm_fn1 = nn.LayerNorm(self.l_size)
        self.norm_fn2 = nn.LayerNorm(self.l_size)
        self.w1 = nn.Linear(self.l_size, self.l_size)
        self.w2 = nn.Linear(self.l_size, self.l_size)

    def forward(self, x):
        y = self.w1(x)
        y = self.nonlin1(y)
        y = self.norm_fn1(y)
        y = self.w2(y)
        y = self.nonlin2(y)
        y = self.norm_fn2(y)
        out = x + y
        return out


class InsidePadMeta(nn.Module):
    def __init__(self, image_size, meta_dim, embed_dim):
        super(InsidePadMeta, self).__init__()
        self.image_size = image_size
        self.base_size = image_size - 2
        self.expand = nn.Linear(1, 4 * image_size - 4, bias=False)
        self.encoder = nn.Sequential(
            nn.Linear(meta_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(embed_dim),
            ResNormLayer(embed_dim),
        ) if meta_dim > 0 else nn.Identity()

    def forward(self, x, meta):
        B, num_token, dim = x.shape
        x = x.transpose(1, 2).view(B, dim, int(sqrt(num_token)), -1)
        base = torch.zeros(B, dim, self.base_size, self.base_size)
        meta = self.expand(meta.unsqueeze(-1)).permute(0, 2, 1)
        meta = self.encoder(meta).permute(0, 2, 1)
        up_down_size = self.image_size
        lr_size = self.base_size
        pad_up, pad_down, pad_left, pad_right = torch.split(meta, dim=-1,
                                                            split_size_or_sections=[up_down_size, up_down_size, lr_size,
                                                                                    lr_size])
        pad_up, pad_down = pad_up.unsqueeze(-2), pad_down.unsqueeze(-2)
        pad_left, pad_right = pad_left.unsqueeze(-1), pad_right.unsqueeze(-1)
        prompt = torch.cat([pad_left, base, pad_right], dim=3)
        prompt = torch.cat([pad_up, prompt, pad_down], dim=2)

        return (prompt + x).flatten(2).transpose(1, 2)
