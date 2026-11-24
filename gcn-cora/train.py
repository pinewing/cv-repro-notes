"""训练脚本：复现论文在 Cora / Citeseer / Pubmed 上的半监督节点分类结果。

用法：
    python train.py --dataset cora --seed 123
    python run_seeds.py --dataset cora --seeds 10
"""

import argparse
import json
import os
import random
import time

import numpy as np
import torch

from data_utils import load_data, to_torch
from model import GCN, masked_accuracy, masked_cross_entropy

PAPER_TEST_ACC = {'cora': 81.5, 'citeseer': 70.3, 'pubmed': 79.0}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='cora', choices=['cora', 'citeseer', 'pubmed'])
    p.add_argument('--data-dir', default=os.path.join(os.path.dirname(__file__), '..', 'gcn', 'data'))
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--lr', type=float, default=0.01)
    p.add_argument('--hidden', type=int, default=16)
    p.add_argument('--dropout', type=float, default=0.5)
    p.add_argument('--weight-decay', type=float, default=5e-4)
    p.add_argument('--early-stopping', type=int, default=10)
    p.add_argument('--seed', type=int, default=123)
    p.add_argument('--device', default='cpu')
    p.add_argument('--log-every', type=int, default=20)
    p.add_argument('--save-json', default='')
    return p.parse_args(argv)


def train_once(args, verbose=True):
    set_seed(args.seed)
    device = torch.device(args.device)

    data = load_data(args.data_dir, args.dataset)
    feats, adj, labels, masks = to_torch(data, device)

    model = GCN(data['num_features'], args.hidden, data['num_classes'], args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    val_losses = []
    best_state, best_val_loss = None, float('inf')
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(feats, adj)

        loss = masked_cross_entropy(logits, labels, masks['train'])
        # 官方只在第一层加 L2（tf.nn.l2_loss = sum(x^2)/2）
        l2 = 0.5 * (model.gc1.weight.pow(2).sum() + model.gc1.bias.pow(2).sum())
        loss = loss + args.weight_decay * l2

        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(feats, adj)
            val_loss = masked_cross_entropy(val_logits, labels, masks['val']).item()
            train_acc = masked_accuracy(val_logits, labels, masks['train'])
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        if verbose and (epoch % args.log_every == 0 or epoch == 1):
            print('epoch {:03d}  loss={:.4f}  train_acc={:.4f}  val_loss={:.4f}  '
                  'elapsed={:.1f}s'.format(epoch, loss.item(), train_acc, val_loss, time.time() - t0))

        # 官方早停：验证损失高于前 N 轮均值即停
        if epoch > args.early_stopping and val_loss > float(np.mean(val_losses[-(args.early_stopping + 1):-1])):
            if verbose:
                print('early stopping at epoch {}, best val_loss={:.4f}'.format(epoch, best_val_loss))
            break

    # 官方做法：用训练结束时的参数评估测试集
    with torch.no_grad():
        test_acc_final = masked_accuracy(model(feats, adj), labels, masks['test'])
        val_acc_final = masked_accuracy(model(feats, adj), labels, masks['val'])

    # 额外记录：验证集最优参数对应的测试集精度
    test_acc_best = None
    if best_state is not None:
        model.load_state_dict(best_state)
        with torch.no_grad():
            test_acc_best = masked_accuracy(model(feats, adj), labels, masks['test'])

    result = {
        'dataset': args.dataset,
        'seed': args.seed,
        'epochs_run': len(val_losses),
        'test_acc_final': round(test_acc_final * 100, 2),
        'test_acc_best_val': round(test_acc_best * 100, 2),
        'val_acc_final': round(val_acc_final * 100, 2),
        'best_val_loss': round(best_val_loss, 6),
        'paper_test_acc': PAPER_TEST_ACC.get(args.dataset),
        'hidden': args.hidden,
        'dropout': args.dropout,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'seconds': round(time.time() - t0, 2),
    }
    if verbose:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main():
    args = build_args()
    result = train_once(args)
    if args.save_json:
        os.makedirs(os.path.dirname(args.save_json), exist_ok=True)
        with open(args.save_json, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print('结果已写入 {}'.format(args.save_json))


if __name__ == '__main__':
    main()
