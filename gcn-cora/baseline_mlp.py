"""对照实验：去掉图结构的 MLP 基线，用来证明 GCN 的增益来自邻接矩阵传播。

论文表 2 中 Cora 上的 MLP 基线为 55.1%，本脚本复刻这一设置。
"""

import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from data_utils import load_data, to_torch
from model import masked_accuracy, masked_cross_entropy
from train import build_args, set_seed


class MLP(nn.Module):
    def __init__(self, num_features, hidden_dim, num_classes, dropout=0.5):
        super().__init__()
        self.dropout = dropout
        self.fc1 = nn.Linear(num_features, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, self.dropout, training=self.training)
        return self.fc2(x)


def main():
    args = build_args()

    set_seed(args.seed)
    data = load_data(args.data_dir, args.dataset)
    feats, _adj, labels, masks = to_torch(data)

    model = MLP(data['num_features'], args.hidden, data['num_classes'], args.dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    val_losses = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(feats)
        loss = masked_cross_entropy(logits, labels, masks['train'])
        l2 = 0.5 * (model.fc1.weight.pow(2).sum() + model.fc1.bias.pow(2).sum())
        loss = loss + args.weight_decay * l2
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = masked_cross_entropy(model(feats), labels, masks['val']).item()
        val_losses.append(val_loss)
        if epoch > args.early_stopping and val_loss > sum(val_losses[-(args.early_stopping + 1):-1]) / args.early_stopping:
            break

    model.eval()
    with torch.no_grad():
        test_acc = masked_accuracy(model(feats), labels, masks['test'])
    result = {'model': 'mlp', 'dataset': args.dataset, 'seed': args.seed,
              'test_acc': round(test_acc * 100, 2), 'paper_mlp_acc': 55.1}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
