"""按节点度数分桶，比较 GCN 与 GAT 的测试精度差异。

动机：GCN 对邻居一视同仁，度数越低可聚合的信息越少；GAT 的注意力可以给
少数邻居更高权重。若 GAT 的提升集中在低度数节点，说明注意力机制起了作用。

用法：
    python analyze_degree.py --dataset cora --seed 0
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from data_utils import load_data
from gat import GAT, build_edges, set_seed
from model import GCN, masked_cross_entropy


def train_gcn(data, epochs=200, lr=0.01, hidden=16, dropout=0.5, wd=5e-4, seed=0):
    set_seed(seed)
    feats = torch.from_numpy(data['features'].toarray()).float()
    labels = torch.from_numpy(data['labels']).float()
    masks = {k: torch.from_numpy(v) for k, v in data['masks'].items()}
    adj = data['adj'].tocoo()
    indices = torch.from_numpy(np.vstack((adj.row, adj.col)).astype(np.int64))
    values = torch.from_numpy(adj.data.astype(np.float32))
    adj_t = torch.sparse_coo_tensor(indices, values, torch.Size(adj.shape)).coalesce()

    model = GCN(data['num_features'], hidden, data['num_classes'], dropout)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        logits = model(feats, adj_t)
        loss = masked_cross_entropy(logits, labels, masks['train'])
        loss = loss + wd * 0.5 * (model.gc1.weight.pow(2).sum() + model.gc1.bias.pow(2).sum())
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        return model(feats, adj_t)


def train_gat(data, epochs=200, lr=0.005, hidden=8, heads=8, dropout=0.6, wd=5e-4, seed=0):
    set_seed(seed)
    feats = torch.from_numpy(data['features'].toarray()).float()
    labels = torch.from_numpy(data['labels']).float()
    masks = {k: torch.from_numpy(v) for k, v in data['masks'].items()}
    edges = build_edges(data['adj_raw'], data['num_nodes'])

    model = GAT(data['num_features'], hidden, data['num_classes'], heads, dropout)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        loss = masked_cross_entropy(model(feats, edges), labels, masks['train'])
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        return model(feats, edges)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='cora')
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    data = load_data(os.path.join(os.path.dirname(__file__), '..', 'gcn', 'data'), args.dataset)
    labels = data['labels'].argmax(1)
    test_idx = np.where(data['masks']['test'])[0]
    degrees = np.asarray((data['adj_raw'] > 0).sum(axis=1)).flatten()

    logits_gcn = train_gcn(data, seed=args.seed)
    logits_gat = train_gat(data, seed=args.seed)
    pred_gcn = logits_gcn.argmax(1).numpy()
    pred_gat = logits_gat.argmax(1).numpy()

    buckets = [(1, 2), (3, 5), (6, 10), (11, 1000)]
    table = []
    for lo, hi in buckets:
        idx = [i for i in test_idx if lo <= degrees[i] <= hi]
        if not idx:
            continue
        acc_gcn = (pred_gcn[idx] == labels[idx]).mean() * 100
        acc_gat = (pred_gat[idx] == labels[idx]).mean() * 100
        table.append({
            'degree': '{}-{}'.format(lo, hi if hi < 1000 else '+'),
            'nodes': len(idx),
            'gcn': round(float(acc_gcn), 2),
            'gat': round(float(acc_gat), 2),
            'delta': round(float(acc_gat - acc_gcn), 2),
        })

    total_gcn = (pred_gcn[test_idx] == labels[test_idx]).mean() * 100
    total_gat = (pred_gat[test_idx] == labels[test_idx]).mean() * 100
    print('总测试精度  GCN={:.2f}%  GAT={:.2f}%'.format(total_gcn, total_gat))
    print('{:<8}{:>8}{:>10}{:>10}{:>10}'.format('度数', '节点数', 'GCN', 'GAT', '差值'))
    for r in table:
        print('{:<8}{:>8}{:>10.2f}{:>10.2f}{:>+10.2f}'.format(r['degree'], r['nodes'], r['gcn'], r['gat'], r['delta']))

    os.makedirs('results', exist_ok=True)
    with open('results/degree_{}.json'.format(args.dataset), 'w', encoding='utf-8') as f:
        json.dump({'dataset': args.dataset, 'seed': args.seed, 'total': {'gcn': round(float(total_gcn), 2),
                                                                        'gat': round(float(total_gat), 2)},
                   'rows': table}, f, ensure_ascii=False, indent=2)
    print('结果写入 results/degree_{}.json'.format(args.dataset))


if __name__ == '__main__':
    main()
