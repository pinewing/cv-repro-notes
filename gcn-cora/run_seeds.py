"""多随机种子实验：输出均值、标准差，并与论文报告值对照。

用法：
    python run_seeds.py --dataset cora --seeds 10
"""

import argparse
import json
import os
import statistics

from train import PAPER_TEST_ACC, build_args, train_once


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='cora', choices=['cora', 'citeseer', 'pubmed'])
    p.add_argument('--seeds', type=int, default=10)
    p.add_argument('--start-seed', type=int, default=0)
    p.add_argument('--out', default='results/seeds_{}.json')
    args = p.parse_args()

    rows = []
    for i in range(args.seeds):
        seed = args.start_seed + i
        argv = ['--dataset', args.dataset, '--seed', str(seed), '--log-every', '200']
        run_args = build_args(argv)
        print('\n===== seed {} ====='.format(seed))
        rows.append(train_once(run_args, verbose=True))

    finals = [r['test_acc_final'] for r in rows]
    bests = [r['test_acc_best_val'] for r in rows]
    summary = {
        'dataset': args.dataset,
        'seeds': rows,
        'final_mean': round(statistics.mean(finals), 2),
        'final_std': round(statistics.pstdev(finals), 2),
        'best_val_mean': round(statistics.mean(bests), 2),
        'best_val_std': round(statistics.pstdev(bests), 2),
        'paper_test_acc': PAPER_TEST_ACC.get(args.dataset),
    }

    out = args.out.format(args.dataset)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print('\n===== 汇总 =====')
    print('数据集        : {}'.format(args.dataset))
    print('论文测试精度  : {}%'.format(summary['paper_test_acc']))
    print('本次(末轮)    : {:.2f} ± {:.2f}%'.format(summary['final_mean'], summary['final_std']))
    print('本次(最优验证): {:.2f} ± {:.2f}%'.format(summary['best_val_mean'], summary['best_val_std']))
    print('明细写入      : {}'.format(out))


if __name__ == '__main__':
    main()
