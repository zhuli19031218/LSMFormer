# -*- encoding:utf-8 -*-
# @Time : 2022/4/30 13:42
# @Author : K
# @File : my_tapnet.py
# @Software :PyCharm
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange
from Anyuan.model.Lstm import *
from Anyuan.model.BiLstm import *
from Anyuan.model.GRU import *
from Anyuan.model.rnn import *
from Anyuan.model.CNN1D import *
from Anyuan.model.CNN2D import *


# helpers

def exists(val):
    return val is not None


def _init_weight(m):
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.2)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)
    # if isinstance(m, nn.Linear):
    #
    #     nn.init.trunc_normal_(m.weight, std=0.2)
    #     # m.weight.data.normal_(0, 0.01)
    #     m.bias.data.zero_()
    # elif isinstance(m, nn.LayerNorm):
    #     nn.init.zeros_(m.bias)
    #     nn.init.ones_(m.weight)


def default(val, d):
    return val if exists(val) else d


# classes

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


# attention

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x, **kwargs):
        print("ffn input", x.shape)
        return self.net(x)


class Attention(nn.Module):
    def __init__(
            self,
            dim,
            heads=8,
            dim_head=16,
            dropout=0.
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)  # 16->>>>16*8*3
        self.to_out = nn.Linear(inner_dim, dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        print(x.shape)
        # print(x[:10])
        h = self.heads
        # print('attension前',x.shape)
        # print('attension前', x.dtype)
        # x=self.to_qkv(x)
        # print('self.to_qkv(x)',x.shape)
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        print('q_pre', q.shape)
        print('v_pre', v.shape)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))
        print('q', q.shape)
        print('v', v.shape)
        sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = sim.softmax(dim=-1)
        attn = self.dropout(attn)
        print('attn', attn.shape)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)', h=h)
        print('out', out.shape)
        print(self.to_out(out).shape)
        return self.to_out(out)


# transformer

class Transformer(nn.Module):
    def __init__(self, dim, in_features, num_tokens, depth, heads, dim_head, attn_dropout, ff_dropout):
        super().__init__()
        num_tokens = 24

        # self.embeds = nn.Embedding(num_tokens, dim)   #每个token使用dim维向量表示
        # self.embeds_2 = nn.Linear(in_features=num_tokens, out_features=dim)
        self.linear = nn.Linear(in_features=in_features, out_features=in_features * dim)
        self.layers = nn.ModuleList([])
        self.in_features = in_features
        self.dim = dim

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=attn_dropout))),
                Residual(PreNorm(dim, FeedForward(dim, dropout=ff_dropout))),  # 前者为做LN的维度，后者为做LN的实体
            ]))

    def forward(self, x):
        # print('x.dtype',x.dtype)

        print('扩充维度前', x.shape)
        # x = np.expand_dims(x, axis=1)
        x = torch.unsqueeze(x, dim=1)

        # print('扩充维度后', x.shape)
        # print(type(x), x.dtype)
        x = self.linear(x.float())
        # print('fc后', x.shape)
        x = x.reshape(-1, self.in_features, self.dim)
        print('重塑后', x.shape)
        # print(x[:10,:,:])

        for attn, ff in self.layers:
            # print('x.shape注意力前', x.shape)
            x = attn(x)
            x = ff(x)
            # print('x.shape注意力后', x.shape)

        return x


class Mlp(nn.Module):

    def __init__(self, in_features, mlp_ratio=4.0, act_layer=nn.ReLU, drop_ration=0.1):
        super().__init__()
        hidden = int(in_features * mlp_ratio)
        self.fc1 = nn.Linear(in_features, hidden)
        self.act = act_layer()
        self.drop = nn.Dropout(drop_ration)
        self.fc2 = nn.Linear(hidden, in_features)
        self.fc3 = nn.Linear(in_features, 2)
        self.drop1 = nn.Dropout(drop_ration)
        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # print('x.shape',x.shape)
        x = self.fc1(x)
        x = self.drop(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        # x = self.drop(x)
        x = self.fc3(x)
        x = self.drop1(x)
        # x = self.sigmoid(x)
        x = self.softmax(x)
        return x


# main class
class MyTabTransformer(nn.Module):
    def __init__(
            self,
            *,
            categories,
            num_continuous,
            dim,
            depth,
            heads,
            dim_head=16,
            dim_out=2,
            mlp_hidden_mults=(4, 2),
            mlp_act=None,
            num_special_tokens=2,
            continuous_mean_std=None,
            attn_dropout=0.,
            ff_dropout=0.,
            num_layers=1
    ):
        super().__init__()
        assert all(map(lambda n: n > 0, categories)), 'number of each category must be positive'  # 特性值必须为正值

        # categories related calculations

        self.num_categories = len(categories)  # 离散特性因子的个数

        self.num_continuous = num_continuous

        # transformer

        self.transformer = Transformer(  # 包含编码
            # num_tokens = total_tokens,
            in_features=self.num_categories,
            num_tokens=24,
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout)

        self.transformer2 = Transformer(  # 包含编码
            # num_tokens=total_tokens,
            in_features=self.num_continuous,
            num_tokens=24,
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout)

        # mlp to logits
        # input_size = (dim * self.num_categories) + num_continuous
        # l = input_size // 8
        # hidden_dimensions = list(map(lambda t: l * t, mlp_hidden_mults))
        # all_dimensions = [input_size, *hidden_dimensions, dim_out]
        # self.mlp = MLP(all_dimensions, act = mlp_act)
        # self.topNet = Mlp(in_features=(dim * self.num_categories) + num_continuous)

        # 参数
        factors = len(categories) * dim + num_continuous

        self.norm = nn.LayerNorm(factors)
        # self.norm = nn.LayerNorm(num_continuous*dim)
        # self.topNet = CNN1D(factors, batch_size=100, device="cuda", drop_ratio=ff_dropout, num_layers=num_layers)
        self.topNet = GRU(factors, batch_size=100, device="cuda", drop_ratio=ff_dropout, num_layers=num_layers)
        # self.topNet = Lstm(factors, batch_size=100, device="cuda", drop_ratio=ff_dropout, num_layers=num_layers)
        # self.apply(_init_weight)

    def forward(self, input):  # input: tensor
        # print(self.num_continuous)
        input = input.squeeze(0)
        x_cont = input[:, :self.num_continuous]
        x_categ = input[:, self.num_continuous:]  # , dtype=torch.int64
        assert x_categ.shape[-1] == self.num_categories, \
            f'you must pass in {self.num_categories} values for your categories input'
        # print('x_categ.shape', x_categ.shape)
        # x_categ = torch.tensor(x_categ, dtype=torch.int64)
        # x_cont = torch.tensor(x_cont, dtype=torch.float32)

        x = self.transformer(x_categ)
        # print('transformer后x_categ.shape', x.shape)
        flat_categ = x.flatten(1)
        # print('拉平后x.shape', flat_categ.shape)
        # print('flat_categ.dtype',flat_categ.dtype)

        assert x_cont.shape[1] == self.num_continuous, \
            f'you must pass in {self.num_continuous} values for your continuous input'

        # x_cont = self.transformer2(x_cont)
        # # print('transformer2后x_cont.shape', x_cont.shape)
        # x_cont = x_cont.flatten(1)
        # print('拉平后x_cont.shape', x_cont.shape)
        normed_cont = x_cont  # self.norm(x_cont)  # layer norm is  useful
        # normed_cont = x_cont
        # print('normed_cont.shape',normed_cont.shape)

        x = torch.cat((flat_categ, normed_cont), dim=-1)
        # print('concat后x.shape',x.shape)
        # print(x.shape)
        x = self.norm(x)
        x = self.topNet(x.unsqueeze(0))

        return x  # .detach().numpy()


if __name__ == '__main__':
    transformer_cfg = {
        'categories': [1] * 66,  # iterable with the number of unique values for categoric feature
        'num_continuous': 10,  # continuous dimensions in data
        'dim': 16,  # hidden dim, paper set at 32
        'dim_out': 2,  # binary prediction
        'depth': 6,  # depth, paper recommended 6
        'heads': 8,  # heads, paper recommends 8
        'attn_dropout': 0.2,  # post-attention dropout
        'ff_dropout': 0.2,  # feed forward dropout
        'mlp_hidden_mults': (4, 2),  # relative multiples of each hidden dimension of the last mlp to logits
        'mlp_act': nn.ReLU(),  # activation for final mlp, defaults to relu
        # 'mlp_act': nn.GELU(),  # activation for final mlp, defaults to relu
        'continuous_mean_std': torch.randn(10, 2)}
    pre_model = MyTabTransformer(**transformer_cfg).to("cuda")  #
    i = torch.randn(1, 10, 76).to("cuda")
    print(pre_model(i).shape)
