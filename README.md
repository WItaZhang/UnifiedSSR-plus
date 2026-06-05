# UnifiedSSR+: An Enhanced Framework for Unified Sequential Search and Recommendation

This repository contains the official implementation of **UnifiedSSR+**, accepted at *ACM Transactions on Intelligent Systems and Technology (TIST)*.

> **UnifiedSSR+: An Enhanced Framework for Unified Sequential Search and Recommendation**  
> Qianqian Zhang, Jiayi Xie, Yikang Zhou, Jing Yi, Zhenzhong Chen  
> Wuhan University

---

## Overview

UnifiedSSR+ jointly models **session-based recommendation** and **session-based search** within a single unified framework. Key innovations include:

- **Intent-oriented session discovery** — automatically segments user interaction histories into intent-coherent sessions via a learnable session partition module.
- **Siamese encoder** — a shared-weight encoder that models product and query sequences symmetrically using multi-head self-attention, cross-attention, and target attention.
- **Multi-task pre-training** — jointly pre-trains on both recommendation and search objectives before task-specific fine-tuning.

---

## Requirements

Python 3.8 is recommended. A conda environment file is provided:

```bash
conda env create -f unifiedssr_env.yml
conda activate unifiedssr
```

Core dependencies: `pytorch=1.13.1`, `numpy=1.24.4`, `pandas=2.0.3`, `scikit-learn=1.3.0`, `matplotlib=3.7.1`, `nltk=3.8.1`, `joblib=1.3.0`.

---

## Data Preparation

Place your dataset under `./data/<dataset_name>/`. Each dataset directory must contain:

- `meta.csv` — tab-separated file with columns: `user_num`, `product_num`, `term_num`, `query_num`
- Pretrain/finetune split files in `.pkl` format (see `data.py` for the expected structure)

Supported dataset names (pass via `--data_name`):

| `--data_name` | Description |
|---|---|
| `JDsearch` | JD.com e-commerce |
| `Amazon_Clothing` | Amazon Clothing, Shoes & Jewelry |
| `Amazon_Electronics_WholeQuery` | Amazon Electronics (whole query) |

---

## Usage

### Dataset-specific Hyperparameters

| Dataset | `--emb_size` | `--enc_num_layer` | `--sub_seq_num` |
|---|---|---|---|
| Amazon-EL | 112 | 1 | 1 |
| Amazon-CL | 128 | 2 | 2 |
| JDsearch | 128 | 3 | 1 |

### 1. Pre-training

```bash
python pretrain.py \
  --data_name Amazon_Electronics_WholeQuery \
  --data_root ./data \
  --tasks search recommendation \
  --emb_size 112 \
  --enc_num_layer 1 \
  --train_batch_size 128 \
  --num_epoch 100
```

Checkpoints are saved to `models/<data_name>/pretrain_<tasks>/<timestamp>/`.

### 2. Fine-tuning

```bash
python finetune.py \
  --data_name Amazon_Electronics_WholeQuery \
  --data_root ./data \
  --tasks search \
  --emb_size 112 \
  --enc_num_layer 1 \
  --trained_model_path models/<data_name>/pretrain_search/<timestamp>/model_100.pth
```

### 3. Evaluation

```bash
python predict.py \
  --data_name Amazon_Electronics_WholeQuery \
  --data_root ./data \
  --tasks search \
  --emb_size 112 \
  --enc_num_layer 1 \
  --trained_model_path models/<data_name>/finetune_search/<timestamp>/model_best.pth
```

<!-- ---



## Citation

If you use this code in your research, please cite:

```bibtex
@article{zhang2026unifiedssr,
  title     = {UnifiedSSR+: An Enhanced Framework for Unified Sequential Search and Recommendation},
  author    = {Zhang, Qianqian and Xie, Jiayi and Zhou, Yikang and Yi, Jing and Chen, Zhenzhong},
  journal   = {ACM Transactions on Intelligent Systems and Technology},
  year      = {2026},
  publisher = {ACM}
}
``` -->

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
