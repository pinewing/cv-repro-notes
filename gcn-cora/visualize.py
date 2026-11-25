"""可视化脚本：生成训练曲线、GAT 注意力热力图、按度数分层的精度柱状图。

用法：
    python visualize.py                 # 生成三张图到 figures/
    python visualize.py --dataset cora --seed 123
"""

import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F

import gat
import train
from data_utils import load_data

FIG_DIR = os.path.join(os.path.dirname(__file__), 'figures')
RES_DIR = os.path.join(os.path.dirname(__file__), 'results')

plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'font.size': 10,
})


def plot_training_curves(hist_gcn, hist_gat, path):
    """左：训练/验证损失曲线；右：训练准确率曲线。两条模型各一条颜色。"""
    e_gcn = [h['epoch'] for h in hist_gcn]
    e_gat = [h['epoch'] for h in hist_gat]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ax = axes[0]
    ax.plot(e_gcn, [h['loss'] for h in hist_gcn], color='#1f77b4', lw=1.6, label='GCN train loss')
    ax.plot(e_gcn, [h['val_loss'] for h in hist_gcn], color='#1f77b4', lw=1.6, ls='--', alpha=0.75,
            label='GCN val loss')
    ax.plot(e_gat, [h['loss'] for h in hist_gat], color='#d62728', lw=1.6, label='GAT train loss')
    ax.plot(e_gat, [h['val_loss'] for h in hist_gat], color='#d62728', lw=1.6, ls='--', alpha=0.75,
            label='GAT val loss')
    ax.set_yscale('log')
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss (log scale)')
    ax.set_title('Training / validation loss (Cora)')
    ax.legend(fontsize=8, ncol=2)

    ax = axes[1]
    ax.plot(e_gcn, [h['train_acc'] * 100 for h in hist_gcn], color='#1f77b4', lw=1.8, label='GCN train acc')
    ax.plot(e_gat, [h['train_acc'] * 100 for h in hist_gat], color='#d62728', lw=1.8, label='GAT train acc')
    ax.set_xlabel('epoch')
    ax.set_ylabel('train accuracy (%)')
    ax.set_title('Training accuracy (Cora, 140 labeled nodes)')
    ax.legend(fontsize=8)
    ax.set_ylim(60, 100)

    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches='tight')
    plt.close(fig)
    print('已生成', path)


def layer_attention(model, x, edges):
    """复算第一层 GAT 的注意力系数，返回 (E, heads) 的 alpha（eval 模式，无 dropout）。"""
    num_nodes = x.size(0)
    h = (x @ model.gat1.weight).view(num_nodes, model.gat1.heads, model.gat1.out_dim)
    src, dst = edges[0], edges[1]
    e_src = (h[src] * model.gat1.att[:, :model.gat1.out_dim]).sum(-1)
    e_dst = (h[dst] * model.gat1.att[:, model.gat1.out_dim:]).sum(-1)
    e = F.leaky_relu(e_src + e_dst, model.gat1.negative_slope)
    node_max = torch.full((num_nodes, model.gat1.heads), float('-inf'))
    node_max = node_max.index_reduce(0, src, e, 'amax', include_self=False)
    e = torch.exp(e - node_max[src])
    denom = torch.zeros(num_nodes, model.gat1.heads).index_add(0, src, e)
    return e / (denom[src] + 1e-16)


def plot_attention_figure(model, x, edges, adj_raw, labels, test_mask, path, top_k=15):
    """左：高度数节点上“邻居 × 头”的相对注意力热力图（按行归一化，便于看谁被选中）；
    右：每个头把注意力分配到 自环 / 低度邻居 / 中度 / 高度 四类的平均份额。"""
    alpha = layer_attention(model, x, edges).detach().numpy()
    src, dst = edges[0].numpy(), edges[1].numpy()
    deg = np.asarray(adj_raw.sum(1)).ravel()
    y = labels.argmax(1) if labels.ndim > 1 else labels

    test_idx = np.where(test_mask)[0]
    high_node = int(test_idx[np.argmax(deg[test_idx])])

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), gridspec_kw={'width_ratios': [1.35, 1]})

    # --- 左：相对注意力热力图 ---
    ax = axes[0]
    m = (src == high_node)
    nb, a = dst[m], alpha[m]
    order = np.argsort(-a.mean(axis=1))[:top_k]
    nb, a = nb[order], a[order]
    uniform = 1.0 / (deg[high_node] + 1.0)     # 完全均匀时每个邻居应得的系数
    rel = a / uniform
    im = ax.imshow(rel, aspect='auto', cmap='RdBu_r', vmin=0, vmax=2)
    ax.set_xticks(range(a.shape[1]))
    ax.set_xticklabels(['h{}'.format(i + 1) for i in range(a.shape[1])])
    ax.set_yticks(range(len(nb)))
    ax.set_yticklabels(['{} (deg {})'.format(int(n), int(deg[n])) for n in nb], fontsize=7)
    ax.set_xlabel('attention head')
    ax.set_title('Center node {} — degree {}, class {}\nattention vs uniform baseline (1.0 = flat, red = boosted)'.format(
        high_node, int(deg[high_node]), int(y[high_node])), fontsize=9)
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02, label='alpha / (1/(deg+1))')

    # --- 右：按邻居度数分类的注意力份额 ---
    ax = axes[1]
    is_self = (src == dst)
    nd = deg[dst]
    buckets = [(is_self, 'self loop'), (nd <= 2, 'low deg 1-2'), ((nd >= 3) & (nd <= 5), 'mid deg 3-5'),
               ((nd >= 6) & (nd <= 10), 'mid-high 6-10'), (nd >= 11, 'high deg 11+')]
    heads = alpha.shape[1]
    xpos = np.arange(heads)
    colors = ['#4d4d4d', '#c6dbef', '#6baed6', '#2171b5', '#08306b']
    boost = (alpha.T * (nd + 1))            # (heads, E) 相对均匀基线的倍数
    width = 0.16
    for i, ((mask_b, name), c) in enumerate(zip(buckets, colors)):
        val = boost[:, mask_b].mean(axis=1) if mask_b.any() else np.zeros(heads)
        ax.bar(xpos + (i - 2) * width, val, width, label=name, color=c)
    ax.axhline(1.0, color='#999', lw=1, ls='--')
    ax.set_xticks(xpos)
    ax.set_xticklabels(['h{}'.format(i + 1) for i in range(heads)])
    ax.set_xlabel('attention head')
    ax.set_ylabel('attention / uniform baseline')
    ax.set_title('Which neighbor degree does each head favor?\n(all edges, Cora; 1.0 = uniform)', fontsize=9)
    ax.legend(fontsize=7, ncol=3, loc='upper center', bbox_to_anchor=(0.5, -0.14), frameon=False)

    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches='tight')
    plt.close(fig)
    print('已生成', path)
    return {'high_degree_node': high_node, 'high_degree': int(deg[high_node]),
            'top_neighbors': [int(n) for n in nb[:5]]}


def plot_degree_buckets(path_json, path_png):
    with open(path_json, encoding='utf-8') as f:
        d = json.load(f)
    labels = ['{} (n={})'.format(r['degree'], r['nodes']) for r in d['rows']]
    gcn = [r['gcn'] for r in d['rows']]
    gat_ = [r['gat'] for r in d['rows']]
    xpos = np.arange(len(labels))
    w = 0.36

    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    b1 = ax.bar(xpos - w / 2, gcn, w, label='GCN', color='#1f77b4')
    b2 = ax.bar(xpos + w / 2, gat_, w, label='GAT', color='#d62728')
    for bars in (b1, b2):
        for b in bars:
            ax.annotate('{:.1f}'.format(b.get_height()), (b.get_x() + b.get_width() / 2, b.get_height()),
                        ha='center', va='bottom', fontsize=8)
    for i, r in enumerate(d['rows']):
        ax.annotate('{:+0.2f}'.format(r['delta']), (xpos[i], min(gcn[i], gat_[i]) - 6),
                    ha='center', fontsize=8, color='#444')
    ax.set_xticks(xpos)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(60, 105)
    ax.set_ylabel('test accuracy (%)')
    ax.set_xlabel('node degree bucket')
    ax.set_title('Accuracy by node degree (Cora, seed 0) — GAT beats GCN only on high-degree nodes')
    ax.legend()
    fig.tight_layout()
    fig.savefig(path_png, dpi=170, bbox_inches='tight')
    plt.close(fig)
    print('已生成', path_png)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='cora')
    p.add_argument('--seed', type=int, default=123)
    p.add_argument('--data-dir', default=os.path.join(os.path.dirname(__file__), '..', 'gcn', 'data'))
    p.add_argument('--edge-dropout', type=float, default=0.0)
    args = p.parse_args()

    os.makedirs(FIG_DIR, exist_ok=True)

    # 1) 训练两条模型并拿到逐 epoch 记录
    gcn_args = train.build_args(['--dataset', args.dataset, '--seed', str(args.seed), '--log-every', '999'])
    gcn_res = train.train_once(gcn_args, verbose=False)
    gat_args = gat.build_args(['--dataset', args.dataset, '--seed', str(args.seed), '--log-every', '999',
                               '--edge-dropout', str(args.edge_dropout)])
    gat_res, model, feats, edges, data = gat.train_once(gat_args, verbose=False, return_model=True)
    print('GCN test {:.2f}%  GAT test {:.2f}%'.format(gcn_res['test_acc_final'], gat_res['test_acc_final']))

    plot_training_curves(gcn_res['history'], gat_res['history'],
                         os.path.join(FIG_DIR, 'training_curves_{}.png'.format(args.dataset)))

    # 2) 注意力热力图：直接用刚训练完的模型权重（best-val 已载入），保证热力图是真实注意力
    labels = data['labels']
    masks = {k: torch.from_numpy(v) for k, v in data['masks'].items()}
    model.eval()
    info = plot_attention_figure(model, feats, edges, data['adj_raw'], labels,
                                 masks['test'].numpy(), os.path.join(FIG_DIR, 'attention_heatmap_{}.png'.format(args.dataset)))

    # 3) 度数分层柱状图（复用已有统计结果）
    deg_json = os.path.join(RES_DIR, 'degree_{}.json'.format(args.dataset))
    if os.path.exists(deg_json):
        plot_degree_buckets(deg_json, os.path.join(FIG_DIR, 'degree_buckets_{}.png'.format(args.dataset)))

    with open(os.path.join(FIG_DIR, 'figures_meta.json'), 'w', encoding='utf-8') as f:
        json.dump({'dataset': args.dataset, 'seed': args.seed,
                   'gcn_test': gcn_res['test_acc_final'], 'gat_test': gat_res['test_acc_final'],
                   'gcn_epochs': gcn_res['epochs_run'], 'gat_epochs': gat_res['epochs_run'],
                   'attention_nodes': info}, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
