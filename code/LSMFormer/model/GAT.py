# -*- encoding=utf-8 -*-
import math
import numpy as np
import scipy.sparse as sp
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from scipy.sparse import coo_matrix


# 2.定义GATConv层
class GATConv(nn.Module):
    def __init__(self, in_channels, out_channels, heads=1, add_self_loops=True, bias=True):
        super(GATConv, self).__init__()
        self.in_channels = in_channels  # 输入图节点的特征数
        self.out_channels = out_channels  # 输出图节点的特征数
        self.adj = None
        self.add_self_loops = add_self_loops

        # 定义参数 θ
        self.weight_w = nn.Parameter(torch.FloatTensor(in_channels, out_channels))
        self.weight_a = nn.Parameter(torch.FloatTensor(out_channels * 2, 1))

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_channels, 1))
        else:
            self.register_parameter('bias', None)

        self.leakyrelu = nn.LeakyReLU()
        self.init_parameters()

    # 初始化可学习参数
    def init_parameters(self):
        nn.init.xavier_uniform_(self.weight_w)
        nn.init.xavier_uniform_(self.weight_a)

        if self.bias != None:
            nn.init.zeros_(self.bias)

    def forward(self, x, edge_index):
        # 1.计算wh，进行节点空间映射
        wh = torch.mm(x, self.weight_w)

        # 2.计算注意力分数
        e = torch.mm(wh, self.weight_a[: self.out_channels]) + torch.matmul(wh, self.weight_a[self.out_channels:]).T

        # 3.激活
        e = self.leakyrelu(e)

        # 4.获取邻接矩阵
        if self.adj == None:
            self.adj = to_dense_adj(edge_index).squeeze()

            # 5.添加自环，考虑自身加权
            if self.add_self_loops:
                self.adj += torch.eye(x.shape[0])

        # 6.获得注意力分数矩阵
        attention = torch.where(self.adj > 0, e, -1e9 * torch.ones_like(e))

        # 7.归一化注意力分数
        attention = F.softmax(attention, dim=1)

        # 8.加权融合特征
        output = torch.mm(attention, wh)

        # 9.添加偏置
        if self.bias != None:
            return output + self.bias.flatten()
        else:
            return output


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self,
                 factors,  # 输入token的dim
                 device,
                 drop_ratio=0.,
                 num_layers=1,
                 nhid=64
                 ):
        super(GCN, self).__init__()
        self.factors = factors
        self.device = device
        self.num_layers = num_layers
        self.backbone1 = nn.LSTM(input_size=factors, hidden_size=4 * factors, num_layers=num_layers)
        self.relu = nn.ReLU()
        self.head = nn.Sequential(
            nn.Linear(in_features=4 * factors, out_features=2 * factors),  #
            nn.Dropout(drop_ratio),
            nn.ReLU(),
            nn.Linear(in_features=2 * factors, out_features=2),
            nn.Dropout(drop_ratio))
            # nn.Softmax(dim=-1)
        self.gc1 = GraphConvolution(factors, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)
        # self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = drop_ratio
        self.activation = nn.Softmax()
        self.fc1 = nn.Linear(nhid, 2)
        self.fc2 = nn.Linear(2, 2)
        # self.lstm = nn.LSTM(nhid, nclass)
        # self.pool = nn.MaxPool2d(3,stride=2)

        self.lstm1 = nn.LSTM(factors, 32, 1)
        self.lstm2 = nn.LSTM(32, 16, 1)
        self.lstm3 = nn.LSTM(16, 2, 1)


    def forward(self, x, adj):
        x1 = F.relu(self.gc1(x, adj))
        x1 = F.dropout(x1, self.dropout, training=self.training)
        x1 = self.gc2(x1, adj)
        x1 = F.dropout(x1, self.dropout, training=self.training)
        x1 = self.gc2(x1, adj)
        x1 = self.fc1(x1)
        x1 = self.fc2(x1)
        # output, _ = self.backbone1(x)
        # # output = self.relu(output)
        # output = self.head(output)
        # output = torch.squeeze(output)
        # print('suqeeze', output.shape)

        return x1


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

if __name__ == '__main__':
    model = GCN(factors=21, device='cpu', drop_ratio=0., num_layers=3)
    input = torch.randn(7092, 21)
    adj = sp.load_npz('../test/adj/load_sparse.npz')
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    output = model(input, adj)
    print(output.shape)
