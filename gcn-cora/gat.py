"""GAT（Graph Attention Networks）复现，用于与 GCN 做同数据对照。

论文：Petar Velickovic et al., "Graph Attention Networks", ICLR 2018, arXiv:1710.10903
官方实现基于 PyTorch Geometric；本文件不依赖 PyG，用稀疏边索引手写掩码自注意力，
按论文 Cora 设置：8 头、每头 8 维、两层、dropout 0.6、lr 0.005、L2 5e-4、Adam、200 epoch。

用法：
    python gat.py --dataset cora --seed 123
    python gat.py --dataset cora --seeds 10
"""

import argparse
import json
import os
import random
import statistics
import time

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from data_utils import load_data, row_normalize
from model import masked_accuracy, masked_cross_entropy

PAPER_GAT = {'cora': 83.0, 'citeseer': 72.5, 'pubmed': 79.0}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_edges(adj_raw, num_nodes):
    """把邻接矩阵（加自环）转成 COO 边索引 [2, E]，方向为 src -> dst。"""
    adj = adj_raw + sp.eye(num_nodes, dtype=np.float32)
    coo = adj.tocoo()
    edges = torch.from_numpy(np.vstack((coo.row, coo.col)).astype(np.int64))
    return edges


class GATLayer(nn.Module):
    """单层图注意力：e_ij = LeakyReLU(a^T [Wh_i || Wh_j])，对邻居做 softmax 后加权聚合。"""

    def __init__(self, in_dim, out_dim, heads=8, dropout=0.6, concat=True, negative_slope=0.2):
        super().__init__()
        self.heads = heads
        self.out_dim = out_dim
        self.concat = concat
        self.dropout = dropout
        self.negative_slope = negative_slope
        self.weight = nn.Parameter(torch.empty(in_dim, heads * out_dim))
        self.att = nn.Parameter(torch.empty(heads, 2 * out_dim))
        nn.init.xavier_uniform_(self.weight)
        nn.init.xavier_uniform_(self.att)

    def forward(self, x, edges):
        num_nodes = x.size(0)
        h = (x @ self.weight).view(num_nodes, self.heads, self.out_dim)
        h = F.dropout(h, self.dropout, training=self.training)

        src, dst = edges[0], edges[1]
        # 每一头的注意力打分
        e_src = (h[src] * self.att[:, :self.out_dim]).sum(-1)
        e_dst = (h[dst] * self.att[:, self.out_dim:]).sum(-1)
        e = F.leaky_relu(e_src + e_dst, self.negative_slope)  # (E, heads)

        # 按源节点做数值稳定的 softmax
        node_max = torch.full((num_nodes, self.heads), float('-inf'), device=x.device)
        node_max = node_max.index_reduce(0, src, e, 'amax', include_self=False)
        e = torch.exp(e - node_max[src])
        denom = torch.zeros(num_nodes, self.heads, device=x.device).index_add(0, src, e)
        alpha = e / (denom[src] + 1e-16)
        alpha = F.dropout(alpha, self.dropout, training=self.training)

        out = torch.zeros(num_nodes, self.heads, self.out_dim, device=x.device)
        out = out.index_add(0, src, alpha.unsqueeze(-1) * h[dst])
        if self.concat:
            return out.reshape(num_nodes, self.heads * self.out_dim)
        return out.mean(dim=1)


class GAT(nn.Module):
    def __init__(self, num_features, hidden_dim, num_classes, heads=8, dropout=0.6, residual=False):
        super().__init__()
        self.dropout = dropout
        self.residual = residual
        self.gat1 = GATLayer(num_features, hidden_dim, heads=heads, dropout=dropout, concat=True)
        self.gat2 = GATLayer(hidden_dim * heads, num_classes, heads=heads, dropout=dropout, concat=False)
        # 残差：第二层输出的维度是类别数，无法与输入直接相加，用一条无图结构的线性投影做跳连
        self.skip = nn.Linear(hidden_dim * heads, num_classes, bias=False) if residual else None

    def forward(self, x, edges):
        x = F.dropout(x, self.dropout, training=self.training)
        h = F.elu(self.gat1(x, edges))
        out = self.gat2(h, edges)
        if self.residual:
            out = out + self.skip(h)
        return out


def build_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='cora', choices=['cora', 'citeseer', 'pubmed'])
    p.add_argument('--data-dir', default=os.path.join(os.path.dirname(__file__), '..', 'gcn', 'data'))
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--lr', type=float, default=0.005)
    p.add_argument('--hidden', type=int, default=8)
    p.add_argument('--heads', type=int, default=8)
    p.add_argument('--dropout', type=float, default=0.6)
    p.add_argument('--weight-decay', type=float, default=5e-4)
    p.add_argument('--edge-dropout', type=float, default=0.0, help='训练时随机丢弃边的比例（自环保留）')
    p.add_argument('--residual', action='store_true', help='第二层加一条线性跳连（非论文结构，用于消融）')
    p.add_argument('--patience', type=int, default=100)
    p.add_argument('--seed', type=int, default=123)
    p.add_argument('--seeds', type=int, default=0, help='>0 时跑多种子并输出均值方差')
    p.add_argument('--log-every', type=int, default=40)
    p.add_argument('--save-json', default='')
    return p.parse_args(argv)


def train_once(args, verbose=True, return_model=False):
    set_seed(args.seed)
    data = load_data(args.data_dir, args.dataset)
    feats = torch.from_numpy(data['features'].toarray()).float()
    labels = torch.from_numpy(data['labels']).float()
    masks = {k: torch.from_numpy(v) for k, v in data['masks'].items()}
    edges = build_edges(data['adj_raw'], data['num_nodes'])

    model = GAT(data['num_features'], args.hidden, data['num_classes'], args.heads, args.dropout,
                residual=args.residual)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    val_losses = []
    best_state, best_val = None, float('inf')
    history = []
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        if args.edge_dropout > 0:
            src, dst = edges[0], edges[1]
            keep = (torch.rand(edges.size(1)) > args.edge_dropout) | (src == dst)
            train_edges = edges[:, keep]
        else:
            train_edges = edges
        loss = masked_cross_entropy(model(feats, train_edges), labels, masks['train'])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = masked_cross_entropy(model(feats, edges), labels, masks['val']).item()
        val_losses.append(val_loss)
        with torch.no_grad():
            train_acc = masked_accuracy(model(feats, edges), labels, masks['train'])
        history.append({'epoch': epoch, 'loss': round(loss.item(), 4),
                        'val_loss': round(val_loss, 4), 'train_acc': round(train_acc, 4)})
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if verbose and (epoch % args.log_every == 0 or epoch == 1):
            print('epoch {:03d}  loss={:.4f}  val_loss={:.4f}  elapsed={:.1f}s'.format(
                epoch, loss.item(), val_loss, time.time() - t0))

        if epoch > args.patience and val_loss > float(np.mean(val_losses[-(args.patience + 1):-1])):
            if verbose:
                print('early stopping at epoch {}'.format(epoch))
            break

    model.eval()
    with torch.no_grad():
        test_final = masked_accuracy(model(feats, edges), labels, masks['test'])
    if best_state is not None:
        model.load_state_dict(best_state)
        with torch.no_grad():
            test_best = masked_accuracy(model(feats, edges), labels, masks['test'])
    else:
        test_best = test_final

    result = {
        'model': 'gat', 'dataset': args.dataset, 'seed': args.seed,
        'heads': args.heads, 'hidden': args.hidden, 'dropout': args.dropout,
        'lr': args.lr, 'epochs_run': len(val_losses),
        'edge_dropout': args.edge_dropout,
        'residual': bool(args.residual),
        'history': history,
        'test_acc_final': round(test_final * 100, 2),
        'test_acc_best_val': round(test_best * 100, 2),
        'paper_test_acc': PAPER_GAT.get(args.dataset),
        'seconds': round(time.time() - t0, 1),
    }
    if verbose:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    if return_model:
        return result, model, feats, edges, data
    return result


def main():
    args = build_args()
    if args.seeds > 0:
        rows = []
        for i in range(args.seeds):
            sub = build_args([
                '--dataset', args.dataset, '--seed', str(i), '--log-every', '400',
                '--epochs', str(args.epochs), '--lr', str(args.lr),
                '--hidden', str(args.hidden), '--heads', str(args.heads),
                '--dropout', str(args.dropout),
                '--weight-decay', str(args.weight_decay),
                '--edge-dropout', str(args.edge_dropout),
                '--patience', str(args.patience),
            ] + (['--residual'] if args.residual else []))
            rows.append(train_once(sub, verbose=False))
        finals = [r['test_acc_final'] for r in rows]
        bests = [r['test_acc_best_val'] for r in rows]
        summary = {
            'model': 'gat', 'dataset': args.dataset, 'seeds': args.seeds,
            'final_mean': round(statistics.mean(finals), 2), 'final_std': round(statistics.pstdev(finals), 2),
            'best_val_mean': round(statistics.mean(bests), 2), 'best_val_std': round(statistics.pstdev(bests), 2),
            'paper_test_acc': PAPER_GAT.get(args.dataset), 'rows': rows,
        }
        out = args.save_json or 'results/gat_{}.json'.format(args.dataset)
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        os.makedirs('results', exist_ok=True)
        with open(out, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print('\n===== GAT 汇总（{} 种子） ====='.format(args.seeds))
        print('论文     : {}%'.format(summary['paper_test_acc']))
        print('末轮     : {:.2f} ± {:.2f}%'.format(summary['final_mean'], summary['final_std']))
        print('最优验证 : {:.2f} ± {:.2f}%'.format(summary['best_val_mean'], summary['best_val_std']))
        print('写入     : {}'.format(out))
    else:
        result = train_once(args)
        if args.save_json:
            with open(args.save_json, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
