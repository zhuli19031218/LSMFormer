# -*- encoding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import sqrt
import math


class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask


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

class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular')
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.value_embedding(x)

        return self.dropout(x)


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
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item()  # c*ln(L_k)
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item()  # c*ln(L_q)

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
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        padding = 1
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=3,
                                  padding=padding,
                                  padding_mode='circular')
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(1, 2)
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
        # x [B, L, D]
        # x = x + self.dropout(self.attention(
        #     x, x, x,
        #     attn_mask = attn_mask
        # ))
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

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


class EncoderStack(nn.Module):
    def __init__(self, encoders, inp_lens):
        super(EncoderStack, self).__init__()
        self.encoders = nn.ModuleList(encoders)
        self.inp_lens = inp_lens

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        # print('stack_x', x.shape)
        x_stack = [];
        attns = []
        for i_len, encoder in zip(self.inp_lens, self.encoders):
            # print('i_len', i_len)
            inp_len = x.shape[1] // (2 ** i_len)
            # print('inp_len', inp_len)
            # x_s1, attn1 = encoder(x)
            # print('x_s1', x_s1.shape)
            x_s, attn = encoder(x[:, -inp_len:, :])
            # x_s, attn = encoder(x)
            # print('x_s', x_s.shape)
            x_stack.append(x_s);
            attns.append(attn)
        x_stack = torch.cat(x_stack, -2)

        return x_stack, attns


class LSTM_attention(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers,
                 dropout,
                 bidirectional,
                 batch_first,
                 classes,
                 device,
                 # pretrained_weight,
                 # update_w2v,
                 ):
        """
        :param input_size: 输入x的特征数,即embedding的size
        :param hidden_size:隐藏层的大小
        :param num_layers:LSTM的层数，可形成多层的堆叠LSTM
        :param dropout: 如果非0，则在除最后一层外的每个LSTM层的输出上引入Dropout层，Dropout概率等于dropout
        :param classes:类别数
        :param batch_first:控制输入与输出的形状，如果为True，则输入和输出张量被提供为(batch, seq, feature)
        :param bidirectional:如果为True，则为双向LSTM
        :param pretrained_weight:预训练的词向量
        :param update_w2v:控制是否更新词向量
        :return:
        """
        super(LSTM_attention, self).__init__()
        # embedding:向量层，将单词索引转为单词向量
        self.device = device
        # encoder层
        self.num_layers = num_layers
        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=batch_first,
            dropout=dropout,
            bidirectional=bidirectional,
        )
        self.encoder2 = nn.LSTM(
            input_size=64,
            hidden_size=64,
            num_layers=1,
            batch_first=batch_first,
            dropout=dropout,
            bidirectional=bidirectional,
        )
        self.encoder3 = nn.LSTM(
            input_size=64,
            hidden_size=64,
            num_layers=1,
            batch_first=batch_first,
            dropout=dropout,
            bidirectional=bidirectional,
        )
        # nn.Parameter:使用这个函数的目的也是想让某些变量在学习的过程中不断的修改其值以达到最优化。
        # self.weight_W = nn.Parameter(torch.Tensor(2 * hidden_size, 2 * hidden_size))
        # self.weight_proj = nn.Parameter(torch.Tensor(2 * hidden_size, 1))
        self.weight_W = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.weight_proj = nn.Parameter(torch.Tensor(hidden_size, 1))
        # 向量初始化
        nn.init.uniform_(self.weight_W, -0.1, 0.1)
        #  在（-0.1，,0.1）的均匀分布中生成值
        nn.init.uniform_(self.weight_proj, -0.1, 0.1)

        # decoder层
        # self.decoder1 = nn.Sequential(
        #     nn.Linear(hidden_size, 2),
        #     nn.Dropout(dropout),
        #     nn.ReLU())
        if bidirectional:
            self.decoder1 = nn.Linear(hidden_size * 2, hidden_size)
            self.dropout = nn.Dropout(dropout)
            self.gelu = nn.GELU()
            self.decoder2 = nn.Linear(hidden_size, classes)
        else:
            self.decoder1 = nn.Linear(hidden_size, hidden_size)
            self.decoder2 = nn.Linear(hidden_size, classes)

    def forward(self, x):
        """
        前向传播
        :param x：输入
        :return:
        """
        # embedding层
        # x.shape=(batch,seq_len);embedding.shape=(num_embeddings, embedding_dim) => emb.shape=(batch,seq_len,embedding_dim)
        # x = torch.unsqueeze(x, dim=1)
        # x = x.transpose(1, 0)
        # encoder层
        # print(x.shape)

        # h0 = torch.randn(self.num_layers * 2, x.shape[0], 512).to('cuda')
        # c0 = torch.randn(self.num_layers * 2, x.shape[0], 512).to('cuda')
        # h0 = torch.randn(self.num_layers * 1, x.shape[0], 64).to(self.device)
        # c0 = torch.randn(self.num_layers * 1, x.shape[0], 64).to(self.device)
        state, hidden = self.encoder(x)
        # print('第一层输出', state.shape)
        # state = state.squeeze(dim=1)
        state, hidden = self.encoder2(state, hidden)
        state, hidden = self.encoder3(state, hidden)
        state = state.transpose(1, 0)
        # print('第二层输出', state2.shape)
        # states: (batch,seq_len, D*hidden_size), D=2 if bidirectional = True else 1, =>[64,75,256]
        # hidden: (h_n, c_n) => h_n / c_n shape:(D∗num_layers, batch, hidden_size) =>[4,64,128]

        # attention:self.weight_proj * tanh(self.weight_W * state)
        # (batch,seq_len, 2*hidden_size) => (batch,seq_len, 2*hidden_size)
        # u = torch.tanh(torch.matmul(state, self.weight_W)).to(self.device)
        u = torch.tanh(torch.matmul(state, self.weight_W))
        # (batch,seq_len, 2*hidden_size) => (batch,seq_len,1)
        # att = torch.matmul(u, self.weight_proj).to(self.device)
        att = torch.matmul(u, self.weight_proj)  # 加性注意力机制
        # print(att.shape)
        att_score = F.softmax(att, dim=1)
        # print(att_score.shape)
        scored_x = state * att_score
        # print(scored_x.shape)
        encoding = torch.sum(scored_x, dim=1)
        # print(encoding.shape)
        # decoder层
        # encoding shape: (batch, D*hidden_size): [64,256]
        #         outputs = self.decoder1(encoding)
        #         x2 = self.decoder1(encoding)
        #         outputs = self.decoder2(outputs)  # outputs shape:(batch, n_class) => [64,2]
        # return outputs, x2
        return encoding


class Mlp(nn.Module):

    def __init__(self, in_features, mlp_ratio=4.0, act_layer=nn.GELU, drop_ration=0.1):
        super().__init__()
        # hidden = int(in_features * mlp_ratio)
        hidden = 64
        self.fc1 = nn.Linear(in_features, hidden)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden, 64)
        self.fc3 = nn.Linear(hidden, 2)
        self.drop = nn.Dropout(drop_ration)
        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # print('x.shape',x.shape)
        x = self.fc1(x)
        x = self.drop(x)
        x = self.act(x)
        x = self.drop(x)
        # x = self.fc2(x)
        # x = self.drop(x)
        x = self.fc3(x)
        x = self.drop(x)
        # x = self.sigmoid(x)
        x = self.softmax(x)
        return x


class InformerStack(nn.Module):
    def __init__(self, c_out, out_len, input_size=21,
                 factor=13, d_model=64, n_heads=8, e_layers=[3, 2, 1], d_layers=2, d_ff=None,
                 dropout=0.4, attn='prob', embed='fixed', freq='h', activation='gelu',
                 output_attention=False, distil=True, mix=True):
        super(InformerStack, self).__init__()
        self.pred_len = out_len
        self.attn = attn
        self.output_attention = output_attention
        self.input_size = input_size
        self.dim = d_model
        # Encoding
        self.linear = nn.Linear(in_features=input_size, out_features=input_size * d_model)
        # self.linear = nn.Linear(in_features=3, out_features=3 * d_model)
        # Attention
        Attn = ProbAttention if attn == 'prob' else FullAttention
        # Encoder

        inp_lens = list(range(len(e_layers)))  # [0,1,2,...] you can customize here
        # print('inp_lens', inp_lens)
        encoders = [
            Encoder(
                [
                    EncoderLayer(
                        AttentionLayer(
                            Attn(False, factor, attention_dropout=dropout, output_attention=output_attention),
                            d_model, n_heads, mix=False),
                        d_model,
                        d_ff,
                        dropout=dropout,
                        activation=activation
                    ) for l in range(el)
                ],
                [
                    ConvLayer(
                        d_model
                    ) for l in range(el - 1)
                ] if distil else None,
                norm_layer=torch.nn.LayerNorm(d_model)
            ) for el in e_layers]
        self.encoder = EncoderStack(encoders, inp_lens)

    def forward(self, input, enc_self_mask=None):
        # input = input[:, 18:]
        x = input.unsqueeze(dim=1)

        x = self.linear(x)

        enc_out = x.reshape(-1, self.input_size, self.dim)
        # enc_out = x.reshape(-1, 3, self.dim)
        # a = x.permute(0, 2, 1)
        # gram = torch.bmm(a, x)
        # enc_out, attns = self.encoder(gram, attn_mask=enc_self_mask)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        # print(enc_out.shape)

        return enc_out


class informerStack_Dlstm(nn.Module):
    def __init__(self, c_out, out_len, input_size, hidden_size, dropout):
        super(informerStack_Dlstm, self).__init__()
        # self.device = device
        self.informer = InformerStack(c_out, out_len)
        self.lstm = LSTM_attention(input_size, hidden_size, num_layers=1, dropout=0.4, bidirectional=False,
                                   batch_first=True, classes=2, device='cuda')
        self.linear = nn.Linear(in_features=1152, out_features=64)
        # self.lstm = BiLstm(factors=64, batch_size=1000, device='cuda', drop_ratio=0.)
        # self.lstm = Lstm(factors=192, batch_size=500, device='cuda', drop_ratio=0.4)
        # self.list = nn.Sequential(
        #     nn.Linear(1152, 256),
        #
        # )
        self.mlp = Mlp(in_features=1088)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input):
        # x = input.unsqueeze(dim=0)
        # input = input.transpose(1, 0)
        # print(input.shape)
        informer_output = self.informer(input)
        # print('informer_output', informer_output.shape)
        informer_output = torch.flatten(informer_output, start_dim=1, end_dim=-1)
        # print(informer_output.shape)
        # print(flatten.shape)
        lstm_output = self.lstm(input)
        # print('lstm_output', lstm_output.shape)
        # lstm_output = torch.flatten(lstm_output, start_dim=1, end_dim=-1)
        output = torch.cat([informer_output, lstm_output], dim=1)
        #         output = self.linear(output)
        # print(output.shape)
        # output = self.softmax(output)
        # output = self.mlp(output)
        # print(output.shape)
        # output = self.mlp(output)
        # return informer_output

        return output

if __name__ == '__main__':
    online_encoder = informerStack_Dlstm(c_out=2, out_len=1, input_size=21, hidden_size=64, dropout=0.)
    online_encoder.load_state_dict(torch.load(r"E:\postgraduate\research_group\实验室数据\pyh_groupSMOTE\informer_LSTM_contrast\pyh_InformerLSTM.pth"))
    online_encoder.to('cuda')
    print(online_encoder(torch.randn(10, 21).to('cuda')).shape)