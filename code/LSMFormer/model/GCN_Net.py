# -*- encoding=utf-8 -*-
import math
import numpy as np
import scipy.sparse as sp
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


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
        print(x.shape)
        x1 = F.relu(self.gc1(x, adj))
        print(x1.shape)
        x1 = F.dropout(x1, self.dropout, training=self.training)
        x1 = self.gc2(x1, adj)
        print(x1.shape)
        x1 = F.dropout(x1, self.dropout, training=self.training)
        x1 = self.gc2(x1, adj)
        print(x1.shape)
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
