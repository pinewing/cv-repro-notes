"""两层 GCN 的 PyTorch 实现，超参与官方 TensorFlow 版本一致。"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConvolution(nn.Module):
    """对应论文式 2：H' = Â H W，Â 为对称归一化邻接矩阵。"""

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        self._reset_parameters()

    def _reset_parameters(self):
        # 官方 inits.glorot：uniform(-sqrt(6/(in+out)), sqrt(6/(in+out)))
        limit = math.sqrt(6.0 / (self.weight.shape[0] + self.weight.shape[1]))
        nn.init.uniform_(self.weight, -limit, limit)

    def forward(self, x, adj):
        support = torch.sparse.mm(adj, x)
        out = support @ self.weight
        if self.bias is not None:
            out = out + self.bias
        return out


class GCN(nn.Module):
    def __init__(self, num_features, hidden_dim, num_classes, dropout=0.5):
        super().__init__()
        self.dropout = dropout
        self.gc1 = GraphConvolution(num_features, hidden_dim)
        self.gc2 = GraphConvolution(hidden_dim, num_classes)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        return self.gc2(x, adj)


def masked_cross_entropy(logits, labels, mask):
    """只在训练/验证/测试掩码内的节点上计算交叉熵。"""
    idx = mask.nonzero(as_tuple=False).view(-1)
    return F.cross_entropy(logits[idx], labels[idx].argmax(dim=1))


@torch.no_grad()
def masked_accuracy(logits, labels, mask):
    idx = mask.nonzero(as_tuple=False).view(-1)
    pred = logits[idx].argmax(dim=1)
    target = labels[idx].argmax(dim=1)
    return (pred == target).float().mean().item()
