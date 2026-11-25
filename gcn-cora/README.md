# GCN 复现（PyTorch 版）

对论文 **Semi-Supervised Classification with Graph Convolutional Networks**（Thomas N. Kipf, Max Welling, ICLR 2017，arXiv:1609.02907）的复现工程。

官方实现（`github.com/tkipf/gcn`，MIT 许可）依赖 TensorFlow 1.15，该版本已无法在现代 Python 上安装，因此本目录用 PyTorch 重写同一套模型与训练流程，并逐项对齐论文设置。

## 环境

| 项目 | 版本 |
| --- | --- |
| Python | 3.12（conda 环境 `gcn`） |
| PyTorch | 2.9.1（CPU 版，`cpu_mkl_py312`） |
| numpy / scipy | 由 conda 官方渠道安装 |
| 硬件 | i7-13620H / 31.6 GB 内存 / Intel UHD 核显（纯 CPU 训练） |

创建环境：

```bash
conda create -n gcn python=3.12 "pytorch=2.9.1=cpu_mkl_py312*" numpy scipy -c defaults -y
conda activate gcn
```

## 运行

```bash
cd reproduction

# 单次训练（默认论文设置：200 epoch，10 轮早停）
python train.py --dataset cora --seed 123

# 多随机种子：均值 ± 标准差
python run_seeds.py --dataset cora --seeds 10

# 对照实验：去掉图结构的 MLP 基线
python baseline_mlp.py --dataset cora --seed 123

# GAT 对照实现（8 头 × 8 维，dropout 0.6，lr 0.005）
python gat.py --dataset cora --seeds 10

# GAT + edge dropout（Citeseer 上最优，10 种子）
python gat.py --dataset citeseer --seeds 10 --edge-dropout 0.7

# 按节点度数分桶比较 GCN / GAT
python analyze_degree.py --dataset cora --seed 0

# 生成三张图：训练曲线、注意力热力图、度数分层柱状图
python visualize.py --dataset cora --seed 123

# 其他数据集
python train.py --dataset citeseer --seed 123
python train.py --dataset pubmed --seed 123
```

## 与论文设置的对齐点

| 论文细节 | 本实现位置 |
| --- | --- |
| 重归一化传播规则 Â = D̃⁻½(A+I)D̃⁻½（式 8） | `data_utils.normalize_adjacency` |
| 特征按行 L1 归一化 | `data_utils.row_normalize` |
| 两层 GCN，隐藏维 16 | `model.GCN`、`--hidden 16` |
| dropout 0.5，作用于每层输入 | `model.GCN.forward` |
| 仅第一层做 L2 正则，系数 5e-4 | `train.train_once`（`tf.nn.l2_loss` = sum/2） |
| Adam，学习率 0.01，最多 200 epoch | `train.build_args` |
| 早停：验证损失超过前 10 轮均值即停 | `train.train_once` |
| Planetoid 公开划分：每类 20 个训练节点、500 验证、1000 测试 | `data_utils.load_data` 的 `masks` |
| Glorot 均匀初始化 | `GraphConvolution._reset_parameters` |

## 实验结果

单位 %，测试集为 Planetoid 公开划分的 1000 个节点。Cora 跑 10 个随机种子，其余为单种子（seed=123）。

| 数据集 | 论文测试精度 | 本实现（末轮） | 本实现（验证集最优） | 差异分析 |
| --- | --- | --- | --- | --- |
| Cora | 81.5 | 80.63 ± 0.93（10 种子） | 81.53 ± 0.70 | 末轮普遍低于最优验证点，说明早停后仍有过拟合 |
| Citeseer | 70.3 | 71.0（单种子） | 71.5 | 与论文一致，官方补丁处理的孤立节点已生效 |
| Pubmed | 79.0 | 78.3（单种子） | 78.5 | 图更大（19717 节点），单次划分波动更大，需多种子 |
| Cora（MLP 基线） | 55.1 | 54.3 | — | 去掉邻接矩阵后掉 26 个点，证明增益来自图传播 |

超参消融（Cora，3 种子，验证集最优口径）：

| hidden \ dropout | 0.3 | 0.5 | 0.7 |
| --- | --- | --- | --- |
| 8 | 80.40 ± 0.59 | 80.67 ± 0.90 | 79.87 ± 1.49 |
| 16 | 81.83 ± 0.31 | 81.77 ± 0.57 | 81.53 ± 1.13 |
| 32 | 81.73 ± 0.25 | 82.03 ± 0.37 | 82.20 ± 0.71 |

结论：隐藏维从 8 提到 16 收益最明显，继续加到 32 只多 0.2–0.4 个点；dropout 0.7 在小隐藏维上明显掉点，在本就稀疏的特征上属于过正则。

## GCN vs GAT 对比

GAT 为手写实现（不依赖 PyG），论文设置 8 头 × 8 维、dropout 0.6、lr 0.005、L2 5e-4：

| 数据集 | GCN（验证最优） | GAT（验证最优） | 论文 GCN / GAT |
| --- | --- | --- | --- |
| Cora | 81.53 ± 0.70（10 种子） | 82.00 ± 0.49（10 种子） | 81.5 / 83.0 |
| Citeseer | 71.5（单种子） | 69.97 ± 0.61（3 种子） | 70.3 / 72.5 |
| Pubmed | 78.64 ± 0.72（5 种子） | 78.77 ± 0.33（3 种子） | 79.0 / 79.0 |

Citeseer 上 GAT 加 edge dropout 0.7 后从 69.97 提到 **71.00 ± 0.25（10 种子）**；同样试过的残差跳连会让 Citeseer 掉到 69.03，已回滚，属无效改动。

度数分桶（Cora，seed 0）：GAT 在 6–10 度节点高 1.7 个点、11+ 度节点高 6.9 个点，但在 1–5 度节点反而低 0.2–0.5 个点。注意力只在邻居足够多、需要区分重要性时才起作用。

可视化（`python visualize.py`，图在 `figures/`）：

- `figures/training_curves_cora.png`：GCN 第 142 轮早停、GAT 跑满 200 轮，训练精度都逼近 100%，GAT 验证损失在 1.1 附近走平
- `figures/attention_heatmap_cora.png`：高度数节点（度 65）上逐邻居的注意力倍数，以及每个头对"自环 / 低度 / 中度 / 高度"邻居的平均偏好（11+ 度邻居约 7–8 倍于均匀基线）
- `figures/degree_buckets_cora.png`：按度数分层的精度柱状图

完整分析见 `../对比报告_GCN_vs_GAT.md`。


## 合规说明

- 数据集 Cora / Citeseer / Pubmed 由官方仓库随代码分发，均为公开学术引文网络数据，不含个人信息
- 代码沿用官方 MIT 许可，重写部分同样以 MIT 发布
- 训练全程离线，不访问任何需要登录或代理的资源
