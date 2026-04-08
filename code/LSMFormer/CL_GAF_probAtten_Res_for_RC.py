# -*- encoding:utf-8 -*-
import torch.nn as nn
import torch
import numpy as np
from math import sqrt
import torch.nn.functional as F

def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)

def bn2d(num_features):
    return nn.BatchNorm2d(num_features)

class GASF(nn.Module):
    def __init__(self, in_feature=21):
        super(GASF, self).__init__()
        # self.bn = nn.BatchNorm1d(in_feature)
        self.factors = in_feature

    def forward(self, x):

        x = (x-x.min())/(x.max()-x.min())*2-1

        x = torch.arccos(x).unsqueeze(1)
        # print(x)
        b = x.permute(0, 2, 1)
        x = x.repeat(1, self.factors, 1)

        b = b.repeat(1, 1, self.factors)

        x = x + b
        x = torch.cos(x)
        return x

class ProbMask():
    def __init__(self, B, H, L, index, scores, device="cpu"):
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[torch.arange(B)[:, None, None],
                    torch.arange(H)[None, :, None],
                    index, :].to(device)
        self._mask = indicator.view(scores.shape).to(device)

    @property
    def mask(self):
        return self._mask


class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask


class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)


class ProbAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):  # n_top: c*ln(L_q)
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(L_K, (L_Q, sample_k))  # real U = U_part(factor*ln(L_k))*L_q
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(-2)

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   M_top, :]  # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))  # factor*ln(L_q)*L_k

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else:  # use mask
            assert (L_Q == L_V)  # requires that L_Q == L_V, i.e. for self-attention only
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1)  # nn.Softmax(dim=-1)(scores)

        context_in[torch.arange(B)[:, None, None],
        torch.arange(H)[None, :, None],
        index, :] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V]) / L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self, queries, keys, values, attn_mask):
        # print(queries.shape, keys.shape, values.shape)
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item()  # c*ln(L_k) np.ceil()向上取整
        # print('U_part', U_part)
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item()  # c*ln(L_q)
        # print('u', u)

        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q

        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)

        # add scale factor
        scale = self.scale or 1. / sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(context, values, scores_top, index, L_Q, attn_mask)

        return context.transpose(2, 1).contiguous(), attn


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads,
                 d_keys=None, d_values=None, mix=False):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.mix = mix

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )
        if self.mix:
            out = out.transpose(2, 1).contiguous()
        out = out.view(B, L, -1)

        return self.out_projection(out), attn


class ConvLayer(nn.Module):
    def __init__(self, c_in=64):
        super(ConvLayer, self).__init__()
        # padding = 1
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=1)
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        # self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = x.transpose(2, 1)
        x = self.downConv(x)
        x = self.norm(x)
        x = self.activation(x)
        # x = self.maxPool(x)
        x = x.transpose(1,2)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        x = x + self.dropout(new_x)
        # print(x.shape)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        # print(y.shape)
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        # print(y.shape)
        return self.norm2(x + y), attn


class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, attn_mask=attn_mask)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns

class InformerStack(nn.Module):
    def __init__(self, dropout=0.3, input_size=23,
                 factor=15, d_model=23, n_heads=8, e=3, d_ff=None,
                 attn='prob', activation='gelu',
                 output_attention=False, distil=False):
        super(InformerStack, self).__init__()
        # self.pred_len = out_len
        self.attn = attn
        self.output_attention = output_attention
        self.input_size = input_size
        self.dim = d_model
        # Encoding
        self.linear = nn.Linear(in_features=input_size, out_features=input_size * d_model)
        # self.linear = nn.Linear(in_features=3, out_features=3 * d_model)
        # Attention
        Attn = ProbAttention
        # Attn = FullAttention
        # Encoder
        self.encoders = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        Attn(False, factor, attention_dropout=dropout, output_attention=output_attention),
                        d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation) for l in range(e)]
            ,[ConvLayer(
                    d_model
                ) for l in range(e)  #  e
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )
    def forward(self, input, enc_self_mask=None):
        enc_out, attns = self.encoders(input, attn_mask=enc_self_mask)
        return enc_out


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, drop_ratio=0.1):
        super(ResidualBlock, self).__init__()
        self.drop_out = nn.Dropout(drop_ratio)
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = bn2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = bn2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.drop_out(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        out = self.drop_out(out)
        return out


class informerStack_ResCNN(nn.Module):
    def __init__(self, in_feature=23, dropout=0.3, cut=None, drop_cnn=0.3):
        super(informerStack_ResCNN, self).__init__()
        # self.device = device
        if cut is None:
            cut = [8, 8, 7]
        self.cut = cut
        self.informer = InformerStack(dropout=dropout, input_size=in_feature, d_model=in_feature)
        self.linear = nn.Linear(in_features=cut[0], out_features=cut[0])
        self.linear1 = nn.Linear(in_features=cut[1], out_features=cut[1])
        self.linear2 = nn.Linear(in_features=cut[2], out_features=cut[2])
        self.gasf = GASF(in_feature=in_feature)
        self.cnn2d = nn.Sequential()
        mid_channels = 16
        self.cnn2d.add_module(f"Residual{0}", self._make_layer(ResidualBlock, 1, mid_channels, 1, drop_cnn=drop_cnn))
        self.cnn2d.add_module(f"Residual{1}", self._make_layer(ResidualBlock, mid_channels, mid_channels*2, 2, drop_cnn=drop_cnn))
        self.head = nn.Sequential(
            nn.BatchNorm1d(mid_channels * 2 * (in_feature // 2 + in_feature % 2) ** 2),
            nn.Linear(in_features=mid_channels * 2 * (in_feature // 2 + in_feature % 2) ** 2, out_features=64),  #
            nn.ReLU(),
            # nn.Dropout(drop_ratio),
            nn.Linear(in_features=64, out_features=2),
        )

    def forward(self, x):
        # x = input.unsqueeze(dim=1)

        # x1 = x[:, :self.cut[0]]
        # x2 = x[:, self.cut[0]:self.cut[0]+self.cut[1]]
        # x3 = x[:, self.cut[0]+self.cut[1]:]
        # mlp_x1 = self.linear(x1)
        # mlp_x2 = self.linear1(x2)
        # mlp_x3 = self.linear2(x3)
        # x = torch.cat([mlp_x1, mlp_x2, mlp_x3], dim=1)
        # x = x.unsqueeze(1)
        # b = x.permute(0,2,1)
        # patch_input = torch.bmm(b, x)

        patch_input = self.gasf(x)
        # print(patch_input)
        output = self.informer(patch_input)
        output = self.cnn2d(output.unsqueeze(1)).flatten(1)
        output = self.head(output)

        return output

    def _make_layer(self, block, inplanes, planes, stride=1, drop_cnn=0.2):
        downsample = None
        if stride != 1 and inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        return block(inplanes, planes, stride, downsample, drop_ratio=drop_cnn)


class informerStack_ResCNN_for_BYOL(nn.Module):
    def __init__(self, in_feature=23, dropout=0.3, cut=None, drop_cnn=0.3):
        super(informerStack_ResCNN_for_BYOL, self).__init__()
        # self.device = device
        if cut is None:
            cut = [8, 8, 7]
        self.cut = cut
        self.informer = InformerStack(dropout=dropout)
        self.gasf = GASF(in_feature=23)
        self.cnn2d = nn.Sequential()
        mid_channels = 16
        self.cnn2d.add_module(f"Residual{0}", self._make_layer(ResidualBlock, 1, mid_channels, 1, drop_cnn=drop_cnn))
        self.cnn2d.add_module(f"Residual{1}", self._make_layer(ResidualBlock, mid_channels, mid_channels*2, 2, drop_cnn=drop_cnn))


    def forward(self, x):
        # x = input.unsqueeze(dim=1)

        # x1 = x[:, :self.cut[0]]
        # x2 = x[:, self.cut[0]:self.cut[0]+self.cut[1]]
        # x3 = x[:, self.cut[0]+self.cut[1]:]
        # mlp_x1 = self.linear(x1)
        # mlp_x2 = self.linear1(x2)
        # mlp_x3 = self.linear2(x3)
        # x = torch.cat([mlp_x1, mlp_x2, mlp_x3], dim=1)
        # x = x.unsqueeze(1)
        # b = x.permute(0,2,1)
        # patch_input = torch.bmm(b, x)

        patch_input = self.gasf(x)
        # print(patch_input)
        output = self.informer(patch_input)
        output = self.cnn2d(output.unsqueeze(1)).flatten(1)
        return output

    def _make_layer(self, block, inplanes, planes, stride=1, drop_cnn=0.2):
        downsample = None
        if stride != 1 and inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        return block(inplanes, planes, stride, downsample, drop_ratio=drop_cnn)

class new_BY(nn.Module):
    def __init__(self, factors, dropout, drop_cnn=0.3):
        super(new_BY, self).__init__()
        # self.online_encoder = informerStack_Dlstm(dropout=0.4)
        self.online_encoder = informerStack_ResCNN(factors, dropout, drop_cnn=drop_cnn)

    def forward(self, x):
        output = self.online_encoder(x)
        return output


import copy
import random
from functools import wraps
from torchvision import models
from torchvision import transforms as T

def default(val, def_val):
    return def_val if val is None else val

def flatten(t):
    return t.reshape(t.shape[0], -1)

def singleton(cache_key):
    def inner_fn(fn):
        @wraps(fn)
        def wrapper(self, *args, **kwargs):
            instance = getattr(self, cache_key)
            if instance is not None:
                return instance

            instance = fn(self, *args, **kwargs)
            setattr(self, cache_key, instance)
            return instance
        return wrapper
    return inner_fn

def get_module_device(module):
    return next(module.parameters()).device

def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val

# loss fn

def loss_fn(x, y):
    # y = y.detach()
    # print(x)
    # print(y)
    x = F.normalize(x, dim=-1, p=2) # 对某一个维度进行L2范式处理
    # print(x)
    y = F.normalize(y, dim=-1, p=2)
    # print(y)
    # print(2 - 2 * (x * y).sum(dim=-1))
    return 2 - 2 * (x * y).sum(dim=-1)

# augmentation utils

class RandomApply(nn.Module):
    def __init__(self, fn, p):
        super().__init__()
        self.fn = fn
        self.p = p
    def forward(self, x):
        if random.random() > self.p:
            return x
        return self.fn(x)

# exponential moving average

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)

# MLP class for projector and predictor

def MLP(dim, projection_size, hidden_size=128):
    return nn.Sequential(
        nn.Linear(dim, hidden_size),  # projection_size自定义的大小为256，和图片的size一致
        nn.BatchNorm1d(hidden_size),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_size, projection_size)
    )

def SimSiamMLP(dim, projection_size, hidden_size=64):
    return nn.Sequential(
        nn.Linear(dim, hidden_size, bias=False),
        nn.BatchNorm1d(hidden_size),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_size, hidden_size, bias=False),
        nn.BatchNorm1d(hidden_size),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_size, projection_size, bias=False),
        nn.BatchNorm1d(projection_size, affine=False)
    )

class BYOL(nn.Module):
    def __init__(
        self,
        # net,
        image_size,
        moving_average_decay = 0.99,
        use_momentum = True
    ):
        super().__init__()

        self.online_encoder = informerStack_ResCNN_for_BYOL(image_size)
        # self.online_encoder = informerStack_Dlstm(c_out=2, out_len=1, input_size=23, hidden_size=64, dropout=0.1)

        self.use_momentum = use_momentum
        self.target_encoder = None
        self.target_ema_updater = EMA(moving_average_decay)  # EMA这种方式，可以有效保持两个网络是不一样的

        self.online_predictor = MLP(4608, 4608, 1024)

    @singleton('target_encoder')
    def _get_target_encoder(self):
        target_encoder = copy.deepcopy(self.online_encoder)
        # target_encoder = copy.deepcopy(self.online_predictor)
        set_requires_grad(target_encoder, False)
        return target_encoder

    def reset_moving_average(self):
        del self.target_encoder
        self.target_encoder = None

    def update_moving_average(self):
        assert self.use_momentum, 'you do not need to update the moving average, since you have turned off momentum for the target encoder'
        assert self.target_encoder is not None, 'target encoder has not been created yet'
        update_moving_average(self.target_ema_updater, self.target_encoder, self.online_encoder)

    def forward(
        self,
        x, y, z,
        return_embedding = False,
        return_projection = True
    ):
        assert not (self.training and x.shape[0] == 1), 'you must have greater than 1 sample when training, due to the batchnorm in the projection layer'

        if return_embedding:
            # print(11)
            return self.online_encoder(x, return_projection=return_projection)
        # print(x, y)

        # image_one, image_two = weak_augment(pd.DataFrame(x.numpy()), 1.1), weak_augment(pd.DataFrame(x.numpy()), 7.5)
        # print(image_one.shape, image_two.shape)
        # y = mask(y, ratios=0.1)
        # online_proj_one, _ = self.online_encoder(x.unsqueeze(1))
        # online_proj_two, _ = self.online_encoder(y.unsqueeze(1))
        online_proj_one = self.online_encoder(x)
        online_proj_two = self.online_encoder(y)
        # print(online_proj_one, online_proj_two)
        online_pred_one = self.online_predictor(online_proj_one)
        online_pred_two = self.online_predictor(online_proj_two)
        # online_pred_one = self.online_predictor(online_proj_one.squeeze(0))
        # online_pred_two = self.online_predictor(online_proj_two.squeeze(0))
        # print(online_pred_one, online_pred_two)
        with torch.no_grad():
            target_encoder = self._get_target_encoder() if self.use_momentum else self.online_encoder
            # target_proj_one, _ = target_encoder(x.unsqueeze(1))
            # target_proj_two, _ = target_encoder(y.unsqueeze(1))
            # target_proj_one = target_encoder(x)
            # target_proj_two = target_encoder(y)
            target_proj_three = target_encoder(z)
            # target_proj_three = target_encoder(z.squeeze(0))
            # print(target_proj_one, target_proj_two)
            # target_proj_one.detach()
            # target_proj_two.detach()
            target_proj_three.detach()

        # loss_one = loss_fn(online_pred_one, target_proj_two.detach())
        # loss_two = loss_fn(online_pred_two, target_proj_one.detach())
        loss_one = loss_fn(online_pred_one, target_proj_three.detach())
        loss_two = loss_fn(online_pred_two, target_proj_three.detach())

        loss = loss_one + loss_two
        return loss.mean()


if __name__ == '__main__':
    print(torch.cuda.is_available())
    model = new_BY(21, 0.3)
    print(model)
    input_data = torch.randn(40, 21)
    output = model(input_data)
    print(output)