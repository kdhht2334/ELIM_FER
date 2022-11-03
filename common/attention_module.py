"""
Inspired by the following: https://github.com/lucidrains/vit-pytorch/
"""
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
#from torchsummary import summary

from torchvision.transforms import Compose, Resize, ToTensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from fabulous.color import fg256


# ViT (vision transformer)
class PatchEmbedding(nn.Module):

    def __init__(self, in_channels: int = 3, patch_size: int = 16, emb_size: int = 768, img_size: int = 224):
        super().__init__()
        self.patch_size = patch_size
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.positions = nn.Parameter(torch.randn((img_size // patch_size)**2 + 1, emb_size))

    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        x = self.projection(x)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        x = torch.cat([cls_tokens, x], dim=1)
        x += self.positions
        return x


class MultiHeadAttention(nn.Module):

    def __init__(self, emb_size: int = 768, num_heads: int = 8, dropout: float = 0):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        # fuse the queries, keys and values in one matrix
        self.qkv = nn.Linear(emb_size, emb_size * 3)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, 2)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        # split keys / queries / values in num_heads
        qkv = rearrange(self.qkv(x), "b n (h d qkv) -> (qkv) b h n d",
                h = self.num_heads, qkv = 3)
        queries, keys, values = qkv[0], qkv[1], qkv[2]
        # sum up over the last axis
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  # batch, num_heads, query_len, key_len
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1/2)
        att = F.softmax(energy, dim=-1) / scaling
        att = self.att_drop(att)
        # sum up over the third axis
        out = torch.einsum('bhal, bhlv -> bhav', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        output = self.projection(out)
        return output


class CrossHeadAttention(nn.Module):

    def __init__(self, emb_size: int = 768, num_heads: int = 8, dropout: float = 0):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.layernorm = nn.LayerNorm(emb_size)
        # fuse the queries, keys and values in one matrix
        self.qkv = nn.Linear(emb_size, emb_size * 3)
        self.att_drop = nn.Dropout(dropout)
        self.bottleneck = nn.Linear(emb_size, emb_size)
        self.projection = nn.Linear(emb_size, 2)
        self.MHA = MultiHeadAttention(emb_size)

    def forward(self, x1: Tensor, x2: Tensor, mask: Tensor = None) -> Tensor:
        # pre-processing
        x1_a = self.layernorm(self.MHA(x1)) + x1
        x2_a = self.layernorm(self.MHA(x2)) + x2
        # split keys / queries / values in num_heads
        qkv1 = rearrange(self.qkv(x1_a), "b n (h d qkv) -> (qkv) b h n d",
                h = self.num_heads, qkv = 3)
        qkv2 = rearrange(self.qkv(x2_a), "b n (h d qkv) -> (qkv) b h n d",
                h = self.num_heads, qkv = 3)
        queries, keys, values = qkv1[0], qkv1[1], qkv2[2]
        # sum up over the last axis
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  # batch, num_heads, query_len, key_len
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1/2)
        att = F.softmax(energy, dim=-1) / scaling
        att = self.att_drop(att)
        # sum up over the third axis
        out = torch.einsum('bhal, bhlv -> bhav', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.bottleneck(out)
        out = self.layernorm(out) + x2_a
        output = self.projection(out).squeeze_()
        return output


class ResidualAdd(nn.Module):

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FeedForwardBlock(nn.Sequential):

    def __init__(self, emb_size: int, expansion: int = 4, drop_p: float = 0.):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class TransformerEncoderBlock(nn.Sequential):

    def __init__(self,
                 emb_size: int = 768,
                 drop_p: float = 0.,
                 forward_expansion: int = 4,
                 forward_drop_p: float = 0.,
                 **kwargs):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, **kwargs),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            ))
        )


# stack blocks
class TransformerEncoder(nn.Sequential):

    def __init__(self, depth: int = 12, **kwargs):
        super().__init__(*[TransformerEncoderBlock(**kwargs) for _ in range(depth)])


# head for specific tasks (classification or regression)
class ClassificationHead(nn.Sequential):

    def __init__(self, emb_size: int = 768, n_classes: int = 1000):
        super().__init__(
                Reduce('b n e -> b e', reduction='mean'),
                nn.LayerNorm(emb_size),
                nn.Linear(emb_size, n_classes))


# summary (ViT)
class ViT(nn.Sequential):

    def __init__(self,
                 in_channels : int = 3,
                 patch_size  : int = 16,
                 emb_size    : int = 384, # 768, 384
                 img_size    : int = 224,
                 depth       : int = 3,   # 12
                 n_classes   : int = 2,   # 1000
                 **kwargs):
        super().__init__(
            PatchEmbedding(in_channels, patch_size, emb_size, img_size),
            TransformerEncoder(depth, emb_size=emb_size, **kwargs),
            ClassificationHead(emb_size, n_classes)
        )


def load_Cross_Attention(emb_size):
    return MultiHeadAttention(emb_size=emb_size)
    #return CrossHeadAttention(emb_size=emb_size)


if __name__ == "__main__":

    from pytorch_model_summary import summary as pt_summary

    MHA = MultiHeadAttention(emb_size=64)
    print(fg256("yellow", pt_summary(MHA, torch.ones_like(torch.empty(10, 1, 64)))))
#    CHA = CrossHeadAttention(emb_size=64)

#    input1 = torch.randn(1, 4, 64)
#    input2 = torch.randn(1, 4, 64)
#    output = CHA(input1, input2)
#    print(fg256("cyan", 'output: ', output.shape))
#    print(fg256("yellow", "CHA", pt_summary(CHA, 
#                        torch.ones_like(torch.empty(10, 4, 64)),
#                        torch.ones_like(torch.empty(10, 4, 64)), show_input=True)))
