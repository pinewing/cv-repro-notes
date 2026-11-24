"""Cora / Citeseer / Pubmed 数据加载，严格对齐 tkipf/gcn 官方实现。

官方仓库用 TensorFlow 1.15 与 maintenance 版 scipy，
这里用 numpy + scipy.sparse 复刻同样的读取、划分与归一化逻辑，
保证输入张量与论文实验设置一致（Planetoid 公开划分）。
"""

import os
import pickle

import numpy as np
import scipy.sparse as sp


def parse_index_file(path):
    with open(path) as f:
        return [int(line.strip()) for line in f]


def row_normalize(features):
    """按行做 L1 归一化（官方 preprocess_features 的做法）。"""
    features = sp.csr_matrix(features, dtype=np.float32)
    rowsum = np.asarray(features.sum(axis=1)).flatten()
    r_inv = np.zeros_like(rowsum, dtype=np.float32)
    nonzero = rowsum > 0
    r_inv[nonzero] = 1.0 / rowsum[nonzero]
    return sp.diags(r_inv).dot(features).tocsr()


def build_adjacency(graph, num_nodes):
    """由邻接字典构建对称邻接矩阵（等价 nx.adjacency_matrix(nx.from_dict_of_lists(graph))）。"""
    rows, cols = [], []
    for node in range(num_nodes):
        for nbr in graph.get(node, []):
            rows.append(node)
            cols.append(nbr)
    data = np.ones(len(rows), dtype=np.float32)
    adj = sp.coo_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes))
    return adj.tocsr()


def normalize_adjacency(adj):
    """对称归一化：D^-1/2 (A+I) D^-1/2（论文式 8 的重归一化技巧）。"""
    adj = sp.coo_matrix(adj)
    rowsum = np.asarray(adj.sum(axis=1)).flatten()
    d_inv_sqrt = np.zeros_like(rowsum, dtype=np.float32)
    nonzero = rowsum > 0
    d_inv_sqrt[nonzero] = np.power(rowsum[nonzero], -0.5)
    d_mat = sp.diags(d_inv_sqrt)
    return (d_mat.dot(adj).dot(d_mat)).tocsr()


def load_data(data_dir, dataset='cora'):
    """返回 features / adjacency / labels / masks，格式与官方 load_data 输出一致。"""
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for name in names:
        with open(os.path.join(data_dir, 'ind.{}.{}'.format(dataset, name)), 'rb') as f:
            objects.append(pickle.load(f, encoding='latin1'))
    x, y, tx, ty, allx, ally, graph = objects

    test_idx_reorder = parse_index_file(os.path.join(data_dir, 'ind.{}.test.index'.format(dataset)))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset == 'citeseer':
        # 官方补丁：citeseer 存在孤立节点，需要把测试集补齐到连续区间
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    features = row_normalize(features.tocsr())

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    num_nodes = features.shape[0]
    adj_raw = build_adjacency(graph, num_nodes)
    adj = normalize_adjacency(adj_raw + sp.eye(num_nodes, dtype=np.float32))

    idx_train = np.arange(len(y))
    idx_val = np.arange(len(y), len(y) + 500)
    idx_test = test_idx_range

    masks = {}
    for split, idx in (('train', idx_train), ('val', idx_val), ('test', idx_test)):
        mask = np.zeros(num_nodes, dtype=bool)
        mask[idx] = True
        masks[split] = mask

    return {
        'features': features,
        'adj': adj,
        'adj_raw': adj_raw,
        'labels': labels.astype(np.float32),
        'masks': masks,
        'num_nodes': num_nodes,
        'num_features': features.shape[1],
        'num_classes': labels.shape[1],
    }


def to_torch(data, device='cpu'):
    """把 scipy 稀疏矩阵转成 torch 稀疏张量。"""
    import torch

    adj = data['adj'].tocoo()
    indices = torch.from_numpy(np.vstack((adj.row, adj.col)).astype(np.int64))
    values = torch.from_numpy(adj.data.astype(np.float32))
    adj_t = torch.sparse_coo_tensor(indices, values, torch.Size(adj.shape)).coalesce().to(device)
    feat_t = torch.from_numpy(data['features'].toarray()).float().to(device)
    labels_t = torch.from_numpy(data['labels']).float().to(device)
    masks_t = {k: torch.from_numpy(v).to(device) for k, v in data['masks'].items()}
    return feat_t, adj_t, labels_t, masks_t
