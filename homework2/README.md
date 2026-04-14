# Homework 2 — YOLOv8n Object Detection Ablation Study

This repository studies how a small set of targeted architecture and training-strategy changes affects YOLOv8n performance on a COCO subset. The goal was to establish a clean baseline, introduce justified improvements, and compare all runs using COCO-style evaluation.

The main experiment artifacts are:

- [`notebooks/01_baseline_eval.ipynb`](notebooks/01_baseline_eval.ipynb)
- [`notebooks/02_arch_only_eval_v2.ipynb`](notebooks/02_arch_only_eval_v2.ipynb)
- [`notebooks/03_train_only_eval_v2.ipynb`](notebooks/03_train_only_eval_v2.ipynb)
- [`notebooks/05_ablation_analysis.ipynb`](notebooks/05_ablation_analysis.ipynb)
- [`configs/baseline.yaml`](configs/baseline.yaml)
- [`configs/arch_only_v2.yaml`](configs/arch_only_v2.yaml)
- [`configs/train_only_v2.yaml`](configs/train_only_v2.yaml)
- [`configs/combined_v2.yaml`](configs/combined_v2.yaml)

## Project goal

Start from a standard YOLOv8n baseline and improve detection performance on a small COCO-style subset by testing:

1. an architectural change,
2. a training-strategy change,
3. a combined version of both.

The final comparison is reported with COCO-style metrics:

- mAP@0.5
- mAP@0.5:0.95
- precision
- recall

## Environment and reproducibility

This project uses Python 3.13+ and depends on Ultralytics, PyTorch, torchvision, pycocotools, matplotlib, pandas, pillow, numpy, jupyter, and ipykernel.

Example setup:

```bash
uv sync
```

or

```bash
pip install -e .
```

The training helper script is:

```bash
uv run python scripts/train_baseline.py --config configs/baseline.yaml --run-name baseline
```

Equivalent runs for the improved variants:

```bash
uv run python scripts/train_baseline.py --config configs/arch_only_v2.yaml --run-name arch_only_v2
uv run python scripts/train_baseline.py --config configs/train_only_v2.yaml --run-name train_only_v2
uv run python scripts/train_baseline.py --config configs/combined_v2.yaml --run-name combined_v2
```

After checkpoints exist, use [`notebooks/05_ablation_analysis.ipynb`](notebooks/05_ablation_analysis.ipynb) to generate the final comparison tables and plots.

## Baseline model setup

The baseline is defined in [`configs/baseline.yaml`](configs/baseline.yaml) and evaluated in [`notebooks/01_baseline_eval.ipynb`](notebooks/01_baseline_eval.ipynb).

### Dataset setup

A 3,000-image subset is sampled from COCO `val2017` and split 80/20 into:

- 2,400 training images
- 600 validation images

The subset is then converted from COCO annotations to YOLO label format.

### Baseline training configuration

| Component | Baseline setting |
|---|---|
| Model | `yolov8n.pt` |
| Transfer learning | Pretrained YOLOv8n weights |
| Epochs | 50 |
| Image size | 640 |
| Batch size | 32 |
| Optimizer | SGD |
| Learning rate | `lr0=0.01`, `lrf=0.01` |
| Scheduler | `cos_lr=false` |
| Momentum | 0.937 |
| Weight decay | 0.0005 |
| Warmup | 3 epochs |
| Augmentations | default YOLO-style HSV jitter, translate=0.1, scale=0.5, fliplr=0.5, mosaic=1.0 |
| Disabled augmentations | `mixup=0.0`, `copy_paste=0.0` |

### Baseline COCO-style evaluation

| Run | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall |
|---|---:|---:|---:|---:|
| baseline | 0.1964 | 0.1355 | 0.2599 | 0.3336 |

## Modified / improved training approach

The improved experiments are split into two clean ablations before being combined.

### 1) Architecture-only v2

References:

- [`notebooks/02_arch_only_eval_v2.ipynb`](notebooks/02_arch_only_eval_v2.ipynb)
- [`configs/arch_only_v2.yaml`](configs/arch_only_v2.yaml)

#### What changed

The main architectural change is:

- `freeze: 10`

This freezes the entire YOLOv8n backbone (layers 0–9) and trains only the neck and detection head.

#### Why this change was made

The baseline fine-tunes the full model, which means all 3.15M parameters are updated on only 2,400 training images. On a dataset this small, that increases the risk of catastrophic forgetting: pretrained COCO features can be degraded by noisy updates from limited data.

Freezing the backbone is intended to:

- preserve the pretrained feature extractor,
- reduce the number of trainable parameters to about 1.08M,
- focus learning capacity on the neck and detection head,
- make transfer learning more stable on limited data.

#### How it affected performance

This change produced a strong improvement over the baseline:

| Run | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | Δ mAP@0.5 vs baseline | Δ mAP@0.5:0.95 vs baseline |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 0.1964 | 0.1355 | 0.2599 | 0.3336 | 0.0000 | 0.0000 |
| arch_only_v2 | 0.2297 | 0.1600 | 0.2716 | 0.3907 | +0.0333 | +0.0245 |

This was the biggest individual improvement in the study.

### 2) Training-only v2

References:

- [`notebooks/03_train_only_eval_v2.ipynb`](notebooks/03_train_only_eval_v2.ipynb)
- [`configs/train_only_v2.yaml`](configs/train_only_v2.yaml)

#### What changed

Compared with the baseline, the training strategy changed as follows:

| Setting | Baseline | v2 |
|---|---:|---:|
| `cos_lr` | `false` | `true` |
| `close_mosaic` | `10` | `20` |
| `hsv_s` | `0.7` | `0.9` |
| `hsv_v` | `0.4` | `0.6` |
| `translate` | `0.1` | `0.15` |
| `scale` | `0.5` | `0.6` |
| `mixup` | `0.0` | `0.0` |
| `copy_paste` | `0.0` | `0.0` |

#### Why these changes were made

The v2 training strategy tries to improve learning without changing the model structure:

- **Cosine LR** smooths learning-rate decay and reduces late-epoch instability.
- **Longer mosaic-off window** gives the model more time to adapt to realistic single-image inputs before evaluation.
- **Stronger HSV jitter** increases color diversity without changing object geometry.
- **Slightly stronger translate/scale jitter** adds robustness to object position and size variation.
- **MixUp and Copy-Paste stay off** because the earlier training-only v1 experiment underperformed, suggesting that those stronger compositing augmentations were not a good fit for this small dataset.

#### How it affected performance

On its own, the training-only v2 update was almost neutral relative to the baseline:

| Run | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | Δ mAP@0.5 vs baseline | Δ mAP@0.5:0.95 vs baseline |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 0.1964 | 0.1355 | 0.2599 | 0.3336 | 0.0000 | 0.0000 |
| train_only_v2 | 0.1934 | 0.1339 | 0.2465 | 0.3241 | -0.0030 | -0.0016 |

So these schedule and augmentation changes did **not** outperform the baseline by themselves, but they became useful when paired with the frozen-backbone architecture.

### 3) Combined v2 (final improved approach)

Reference:

- [`configs/combined_v2.yaml`](configs/combined_v2.yaml)

#### What changed

The final model combines both successful ideas:

- frozen backbone (`freeze: 10`), and
- the v2 training strategy (`cos_lr=true`, `close_mosaic=20`, stronger HSV jitter, slightly stronger translate/scale jitter).

#### Why this combined approach makes sense

The two ideas are complementary:

- the frozen backbone preserves strong pretrained features,
- the training updates help the unfrozen neck and head adapt more smoothly and more realistically,
- the model spends more of late training on single-image inputs that better match evaluation conditions.

#### How it affected performance

This was the best run overall:

| Run | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | Δ mAP@0.5 vs baseline | Δ mAP@0.5:0.95 vs baseline |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 0.1964 | 0.1355 | 0.2599 | 0.3336 | 0.0000 | 0.0000 |
| combined_v2 | 0.2319 | 0.1618 | 0.2754 | 0.3923 | +0.0355 | +0.0263 |

This means the final model improved over the baseline by:

- **+0.0355 mAP@0.5**
- **+0.0263 mAP@0.5:0.95**
- **+0.0155 precision**
- **+0.0587 recall**

## COCO-style evaluation comparison before and after the changes

The final comparison comes from [`notebooks/05_ablation_analysis.ipynb`](notebooks/05_ablation_analysis.ipynb).

### Full ablation table

| Run | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | Δ mAP@0.5 vs baseline | Δ mAP@0.5:0.95 vs baseline |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 0.1964 | 0.1355 | 0.2599 | 0.3336 | +0.0000 | +0.0000 |
| arch_only | 0.0540 | 0.0329 | 0.2339 | 0.0900 | -0.1424 | -0.1026 |
| train_only | 0.1219 | 0.0811 | 0.1867 | 0.2168 | -0.0745 | -0.0544 |
| arch_only_v2 | 0.2297 | 0.1600 | 0.2716 | 0.3907 | +0.0333 | +0.0245 |
| train_only_v2 | 0.1934 | 0.1339 | 0.2465 | 0.3241 | -0.0030 | -0.0016 |
| combined_v2 | 0.2319 | 0.1618 | 0.2754 | 0.3923 | +0.0355 | +0.0263 |

### Before vs. after summary

| Model | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall |
|---|---:|---:|---:|---:|
| Baseline | 0.1964 | 0.1355 | 0.2599 | 0.3336 |
| Final improved model (`combined_v2`) | 0.2319 | 0.1618 | 0.2754 | 0.3923 |

## Interpretation

The experiment results suggest three clear conclusions:

1. **Freezing the backbone was the most effective single change.**  
   `arch_only_v2` delivered a clear improvement over the baseline and explains most of the final gain.

2. **Training-only v2 changes were not enough on their own.**  
   The schedule and augmentation updates were reasonable, but by themselves they were slightly below baseline.

3. **The combined approach worked best.**  
   `combined_v2` achieved the highest mAP, precision, and recall, which means the architectural change and the training-strategy change are complementary.

## Result artifacts

The repository already includes generated comparison artifacts in [`results/`](results/):

- [`results/summary_table.csv`](results/summary_table.csv)
- [`results/map_comparison.png`](results/map_comparison.png)
- [`results/all_training_curves.png`](results/all_training_curves.png)
- [`results/precision_recall_scatter.png`](results/precision_recall_scatter.png)
- [`results/side_by_side_predictions.png`](results/side_by_side_predictions.png)

## Final conclusion

The best-performing configuration in this repository is **`combined_v2`**, which combines a frozen YOLOv8n backbone with a more conservative small-dataset training strategy. Compared with the baseline, it gives the best COCO-style evaluation numbers and shows that, for this task, preserving pretrained backbone features mattered more than aggressive end-to-end fine-tuning.
