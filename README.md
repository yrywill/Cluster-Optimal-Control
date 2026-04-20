# Cluster-Based Data Selection for Continual Pre-training

通过 KMeans 聚类训练数据，然后用 PMP (Perturbation-based Meta-Policy) 根据验证集梯度动态调整各聚类的采样权重，让训练数据分布在训练过程中自适应地向有利于 dev loss 的方向倾斜。

## 1. 快速开始

```bash
# 依赖
pip install -r requirements.txt

# 一键启动（8 卡 DDP，默认配置）
bash launch_train.sh

# 自定义 GPU 数 / 输出目录 / 端口
bash launch_train.sh 4
OUT_DIR=outputs/my_run  bash launch_train.sh
MASTER_PORT=29501       bash launch_train.sh
```

等价的手动命令：
```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
torchrun --nproc_per_node=8 --master_port=29500 train.py \
    --config configs/default.yaml \
    training.save_dir=outputs/run_$(date +%Y%m%d_%H%M%S)
```

**所有超参都在 `configs/default.yaml`**，CLI 只用来覆盖输出目录 / GPU 数 / 端口之类的启动器参数。

## 2. 项目结构

```
cluster_data_selection/
├── train.py                       # 入口
├── launch_train.sh                # 8-GPU DDP 启动脚本（推荐入口）
├── configs/
│   ├── default.yaml               # 全部超参
│   ├── default_with_early_exit.yaml
│   ├── ds_zero1.json              # DeepSpeed ZeRO-1 (推荐)
│   ├── ds_zero2.json
│   ├── ds_zero3.json              # ZeRO-3 纯 GPU
│   └── ds_zero3_offload.json      # ZeRO-3 + CPU offload
├── data/
│   ├── json_dataset.py            # 通用 JSON/JSONL loader
│   ├── eval_dataset.py            # Few-shot MCQ 评估
│   └── cluster_dataset.py         # ClusterDataset + ClusterWeightedSampler
├── clustering/
│   ├── kmeans_clusterer.py        # MiniBatch / 完整 KMeans / Faiss
│   ├── random_clusterer.py        # 随机聚类 baseline
│   └── early_exit_kmeans.py       # early-exit 特征 KMeans
├── pmp/
│   ├── count_sketch.py            # CountSketch 投影（O(1) 内存）
│   ├── grad_utils_sketch.py       # 当前默认路径：sketch 快路径（rank 切片）
│   ├── grad_utils.py              # 传统 JVP 路径（保留，不再走）
│   ├── model_wrapper.py           # 参数向量化辅助
│   └── projection.py              # Rademacher / Gaussian 投影
├── trainer/
│   ├── integrated_trainer.py      # IntegratedClusterTrainer — 训练主循环
│   └── ring_buffer.py             # 近期参数快照环形缓冲
├── utils/
│   ├── config.py                  # OmegaConf 加载
│   └── layer_access.py
├── dataset-100k/                  # 训练数据（*.json / *.jsonl）
└── valid/                         # 验证集（MMLU）
```

## 3. 核心流程

```
                         ┌── dev grad (sketch) ──┐
                         ▼                        │
训练数据 → KMeans 聚类 → ClusterSampler → LM forward/backward
                ▲                                 │
                │         每 update_interval 步    │
                └── 权重更新 ←─── grad_γ = Σ <dev, cluster>
```

1. **聚类**：启动时用 embedding model（默认 qwen2.5-0.5B 取 intermediate 层）对 111K 样本做 MiniBatch KMeans（默认 K = N/cluster_size = 222），产出 `cluster_all.json` + `cluster_assignments_initial.json`
2. **训练**：`ClusterWeightedSampler` 按当前权重采样 → Llama-3.2-3B forward/backward + AdamW
3. **PMP 更新**（每 `pmp.update_interval` 步）：
   - 算 `q = sketch(∇L_dev)`（各 rank 分 dev batches，`all_reduce`）
   - 对 `K // world_size` 个簇（按 rank 切片）算 `v_k = sketch(∇L_k)`，`ct_k = pmp.lr · <q, v_k>`
   - `all_reduce(SUM)` 合并所有 rank 的 `grad_γ`
   - `softmax(grad_γ / temperature)` → 新权重
   - 连续 `drop_patience` 次负贡献的簇会被永久丢弃

> 当前默认走的是 `grad_utils_sketch.py` 的 **CountSketch + rank 切片**快路径，约 2.5× 快于传统 JVP + 8× 快于每 rank 重复跑。

## 4. 关键配置项（`configs/default.yaml`）

| 配置路径 | 默认值 | 说明 |
|---|---|---|
| `model.path` | `llama-3.2-3B` | HuggingFace 路径 |
| `model.max_length` | **1024** | 训练 seq length |
| `model.attn_impl` | `flash_attention_2` | 注意力实现 |
| `model.gradient_checkpointing` | true | 省激活显存 |
| `data.train_dir` | `dataset-100k` | 训练数据目录 |
| `data.dev_dir` | `valid` | dev 目录（MMLU） |
| `data.n_shot` | 3 | Few-shot 示例数 |
| `clustering.method` | `minibatch` | minibatch / kmeans / faiss / random |
| `clustering.cluster_size` | 500 | 目标每簇样本数；K = N/size |
| `clustering.kmeans.feature` | `intermediate` | projection / embedding / ghost / intermediate |
| `clustering.embedding_model.path` | `qwen2.5-0.5B` | 小模型特征抽取（比主模型快很多） |
| `pmp.update_interval` | 50 | 每 50 步做一次 PMP |
| `pmp.lr` | **0.3** | PMP 内部 lr（越大权重变化越快） |
| `pmp.temperature` | **0.3** | softmax 温度（越大越均匀） |
| `pmp.drop_patience` | 5 | 连续 N 次负贡献才丢 |
| `pmp.ghost_ip.enabled` | true | 启用 CountSketch 快路径 |
| `pmp.ghost_ip.proj_dim` | 8192 | sketch 维度 |
| `training.total_iters` | 500 | 训练步数 |
| `training.batch_size` | 4 | 单卡 micro batch |
| `training.gradient_accumulation_steps` | 2 | 等效 batch = 4 × 2 × ngpu |
| `training.lr` | 3e-5 | 主模型 lr |
| `training.eval_interval` | 100 | 每 N 步评估 |

## 5. 启动模式

### DDP（当前默认）

`configs/default.yaml` 里 `deepspeed.enabled: false`：

```bash
bash launch_train.sh                         # 8 GPU
bash launch_train.sh 4                       # 4 GPU
```

### DeepSpeed ZeRO-3（大模型）

```bash
deepspeed --num_gpus=8 train.py \
    --config configs/default.yaml \
    deepspeed.enabled=true
```

显存紧张时换 offload 版：
```bash
deepspeed --num_gpus=8 train.py \
    --config configs/default.yaml \
    deepspeed.enabled=true \
    deepspeed.config_file=configs/ds_zero3_offload.json
```

### CLI 覆盖任意参数（OmegaConf dot-list）

```bash
torchrun --nproc_per_node=8 train.py --config configs/default.yaml \
    model.path=Qwen/Qwen2-7B \
    training.lr=1e-5 \
    clustering.method=random \
    pmp.lr=0.5
```

## 6. 评估

三种模式，`data.eval_format` 切换：

| 模式 | 行为 | 输出 |
|---|---|---|
| `fewshot` (n_shot>0) | 拼接 N 个同类示例 → 预测 A/B/C/D | `fewshot_acc` + `{domain}_loss` + `{domain}_ppl` |
| `fewshot` (n_shot=0) | 直接 MCQ | 同上 |
| `text` | 纯 next-token prediction loss | 仅 `{domain}_loss / ppl` |

多领域 dev 通过 `data.dev_domains` 配置：
```yaml
data:
  dev_domains:
    - { name: math,    dir: data/dev/math,    weight: 0.5 }
    - { name: code,    dir: data/dev/code,    weight: 0.3 }
    - { name: general, dir: data/dev/general, weight: 0.2 }
```

## 7. 聚类方法

| `clustering.method` | 说明 | 适用 |
|---|---|---|
| `minibatch`（默认） | sklearn MiniBatchKMeans | 通用 |
| `kmeans` | sklearn 完整 KMeans | 小数据、追求精度 |
| `faiss` | Faiss GPU KMeans | 百万级以上，需 `faiss-gpu` |
| `random` | 随机分配 | Baseline 对比 |

特征类型 `clustering.kmeans.feature`：

- `intermediate`（默认） — 主模型中间层 hidden states（`embed_layer=-1` = 中间）
- `embedding` — 最后一层 hidden mean（慢，保留所有层）
- `projection` — 样本 LM 梯度随机投影到 `proj_dim`
- `ghost` — projection + 参数掩码

## 8. CountSketch 快路径（强烈推荐）

```yaml
pmp:
  ghost_ip:
    enabled: true       # O(1) sketch，跳过 ring-buffer 遍历
    proj_dim: 8192
    proj_type: count_sketch
```

相比传统 JVP，**内存 16 GB → 60 MB**，速度 8× 起（rank 切片后）。
> 代码路径：`pmp/grad_utils_sketch.py:compute_cluster_contributions_sketch`

## 9. 输出与 checkpoint

每个 run 落到 `outputs/run_<timestamp>/`：

```
run_20260420_185047/
├── train.log                           # 全部日志
├── cluster_all.json                    # 所有簇 metadata
├── cluster_assignments_initial.json    # 初始簇分配
├── cluster_assignments_latest.json     # 含 drop 之后
├── cluster_weight_history.jsonl        # 每次 PMP 后的权重
├── grad_gamma_<step>.pt                # 每次 PMP 的 grad_γ
└── step_<step>/                        # HuggingFace-compatible checkpoint
    ├── config.json
    ├── model.safetensors
    ├── tokenizer*.json
    └── ...
```

## 10. 关键注意点

- **显存预算**（单卡 95 GB H20，`max_length=1024`）：
  - 训练态 ~16 GB（bf16 + AdamW + checkpointing）
  - PMP 峰值 ~50 GB（含 dev sketch + cluster forward）
  - 若 seq 升到 2048，PMP 峰值会摸到 95 GB 边缘，容易 OOM
- **`pmp.lr + pmp.temperature`** 联动决定权重变化幅度。常见参考值：
  - **0.3 / 0.3**（当前推荐）— 温和
  - 1.0 / 0.1（旧默认）— 激进，最大权重可到 0.33
- **`drop_bad_clusters=true + drop_patience=5`** 会在 5 次 PMP 后永久丢连续负贡献的簇，可能丢到 20+ 个
- **DDP 下** PMP 做了 rank 切片：每 rank 算 `clusters[rank::world_size]` ~ K/world_size 个簇，最后 `all_reduce(SUM)` 合并
- **DeepSpeed ZeRO-3** 下走 `deepspeed.zero.GatheredParameters` 汇聚参数后做 PMP

## 11. 常见问题

| 现象 | 原因 | 对策 |
|---|---|---|
| PMP 阶段 CUDA OOM | PMP 追加激活 + logits 大 | 降 `model.max_length` 或开 ZeRO / offload |
| PMP 极慢（每次 >10 min） | 老代码每 rank 都算全量 | 已修复，见 `grad_utils_sketch.py:137` rank 切片 |
| 权重全部集中在少数簇 | `pmp.lr` 太大 / `temperature` 太小 | 调到 0.3/0.3 或更小 |
| `dropped` 过多 | `drop_patience` 太小 | 调到 10 或禁用 `drop_bad_clusters` |
| fewshot_acc 不稳 | dev 只有 100 条，噪声大 | 加大 `data.dev_num` |

## 12. 依赖

```bash
pip install -r requirements.txt
```

主要依赖：`torch`, `transformers`, `omegaconf`, `scikit-learn`, `deepspeed`(可选), `faiss-gpu`(可选), `flash-attn`(可选)。
