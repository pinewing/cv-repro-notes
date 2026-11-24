"""消融实验：隐藏维与 dropout 对 Cora 节点分类的影响。

用法：
    python ablation.py --seeds 3
"""

import argparse
import itertools
import json
import os
import statistics

from train import build_args, train_once


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='cora')
    p.add_argument('--seeds', type=int, default=3)
    p.add_argument('--hidden', default='8,16,32')
    p.add_argument('--dropout', default='0.3,0.5,0.7')
    p.add_argument('--out', default='results/ablation_{}.json')
    args = p.parse_args()

    hiddens = [int(x) for x in args.hidden.split(',')]
    dropouts = [float(x) for x in args.dropout.split(',')]

    table = []
    for hidden, dropout in itertools.product(hiddens, dropouts):
        finals, bests = [], []
        for seed in range(args.seeds):
            argv = ['--dataset', args.dataset, '--hidden', str(hidden), '--dropout', str(dropout),
                    '--seed', str(seed), '--log-every', '200']
            r = train_once(build_args(argv), verbose=False)
            finals.append(r['test_acc_final'])
            bests.append(r['test_acc_best_val'])
        row = {
            'hidden': hidden,
            'dropout': dropout,
            'final_mean': round(statistics.mean(finals), 2),
            'final_std': round(statistics.pstdev(finals), 2),
            'best_mean': round(statistics.mean(bests), 2),
            'best_std': round(statistics.pstdev(bests), 2),
        }
        table.append(row)
        print('hidden={:<3d} dropout={:<4.1f}  末轮={:5.2f}±{:.2f}  最优验证={:5.2f}±{:.2f}'.format(
            hidden, dropout, row['final_mean'], row['final_std'], row['best_mean'], row['best_std']))

    out = args.out.format(args.dataset)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w', encoding='utf-8') as f:
        json.dump({'dataset': args.dataset, 'seeds': args.seeds, 'rows': table}, f, ensure_ascii=False, indent=2)
    print('结果写入 {}'.format(out))


if __name__ == '__main__':
    main()
