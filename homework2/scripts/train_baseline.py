"""
scripts/train.py
----------------
Trains a single YOLOv8 run from a given config YAML and saves metrics + plots.
Converted from 01_baseline_eval.ipynb — works identically for baseline,
arch_only, train_only, and combined runs by swapping --config.

Usage:
    uv run python scripts/train.py --config configs/baseline.yaml

Full example with all flags:
    uv run python scripts/train.py \\
        --config   configs/baseline.yaml \\
        --data     data/coco_subset.yaml \\
        --out      results/baseline \\
        --run-name baseline \\
        --skip-subset \\
        --skip-labels \\
        --skip-train \\
        --checkpoint runs/baseline/weights/best.pt

tmux workflow:
    tmux new -s baseline
    uv run python scripts/train.py --config configs/baseline.yaml
    Ctrl+B then D  ← detach (training keeps running)
    tmux attach -t baseline  ← reattach later
"""

import argparse
import json
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # no GUI — renders PNGs directly, required for tmux
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Train a YOLOv8 run and save metrics + plots.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--config", required=True,
        help="Path to Ultralytics training config YAML (e.g. configs/baseline.yaml)",
    )
    p.add_argument(
        "--data", default=None,
        help="Path to dataset YAML. Overrides any 'data:' key in --config. "
             "Defaults to data/coco_subset.yaml relative to project root.",
    )
    p.add_argument(
        "--out", default=None,
        help="Directory for metrics JSON and plots. "
             "Defaults to results/<run-name>.",
    )
    p.add_argument(
        "--run-name", default=None,
        help="Name of the run (used for Ultralytics project/name and default --out). "
             "Defaults to the stem of --config (e.g. 'baseline').",
    )
    p.add_argument(
        "--n-subset", type=int, default=3000,
        help="Total number of images to sample from val2017 for the subset. "
             "Ignored if --skip-subset is set.",
    )
    p.add_argument(
        "--train-split", type=float, default=0.8,
        help="Fraction of --n-subset used for training (rest is val).",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for subset sampling.",
    )
    p.add_argument(
        "--conf", type=float, default=0.25,
        help="Confidence threshold for sample prediction visualizations.",
    )
    p.add_argument(
        "--n-sample", type=int, default=200,
        help="Number of val images for confidence distribution plot.",
    )
    p.add_argument(
        "--skip-subset", action="store_true",
        help="Skip subset creation (images already copied).",
    )
    p.add_argument(
        "--skip-labels", action="store_true",
        help="Skip COCO→YOLO label conversion (labels already exist).",
    )
    p.add_argument(
        "--skip-train", action="store_true",
        help="Skip training and load --checkpoint instead.",
    )
    p.add_argument(
        "--checkpoint", default=None,
        help="Path to existing best.pt. Implies --skip-train.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def resolve_paths(args):
    """Return a dict of all resolved absolute paths used throughout the script."""
    # Project root = parent of scripts/
    project_root = Path(__file__).resolve().parent.parent
    run_name     = args.run_name or Path(args.config).stem

    data_yaml    = Path(args.data).resolve() if args.data \
                   else project_root / "data" / "coco_subset.yaml"
    config_yaml  = Path(args.config).resolve()
    out_dir      = Path(args.out).resolve() if args.out \
                   else project_root / "results" / run_name

    coco_root  = project_root / "data"
    subset_dir = project_root / "data" / "coco_subset"

    out_dir.mkdir(parents=True, exist_ok=True)

    return {
        "project_root": project_root,
        "run_name":     run_name,
        "config_yaml":  config_yaml,
        "data_yaml":    data_yaml,
        "out_dir":      out_dir,
        "coco_root":    coco_root,
        "subset_dir":   subset_dir,
        "val_images":   subset_dir / "images" / "val",
        "runs_dir":     project_root / "runs",
    }


# ---------------------------------------------------------------------------
# Step 1 — build COCO subset
# ---------------------------------------------------------------------------

def build_subset(paths, n_images, train_split, seed):
    coco_root  = paths["coco_root"]
    subset_dir = paths["subset_dir"]

    print(f"\n[subset] Sampling {n_images} images from val2017 (seed={seed}) ...")
    random.seed(seed)

    ann_path = coco_root / "annotations" / "instances_val2017.json"
    if not ann_path.exists():
        sys.exit(f"[subset] ERROR: {ann_path} not found. Download COCO annotations first.")

    with open(ann_path) as f:
        data = json.load(f)

    imgs = data["images"].copy()
    random.shuffle(imgs)
    n_train     = int(n_images * train_split)
    train_imgs  = imgs[:n_train]
    val_imgs    = imgs[n_train:n_images]
    train_ids   = {img["id"] for img in train_imgs}
    val_ids     = {img["id"] for img in val_imgs}

    def write_split(subset_imgs, img_ids, split_name):
        img_out = subset_dir / "images" / split_name
        ann_out = subset_dir / "annotations"
        img_out.mkdir(parents=True, exist_ok=True)
        ann_out.mkdir(parents=True, exist_ok=True)

        src_dir = coco_root / "images" / "val2017"
        for img in subset_imgs:
            src = src_dir / img["file_name"]
            dst = img_out / img["file_name"]
            if not dst.exists():
                shutil.copy(src, dst)

        anns = [a for a in data["annotations"] if a["image_id"] in img_ids]
        out  = {
            "images":      subset_imgs,
            "annotations": anns,
            "categories":  data["categories"],
        }
        with open(ann_out / f"instances_{split_name}.json", "w") as f:
            json.dump(out, f)

        print(f"[subset]   {split_name}: {len(subset_imgs)} images, {len(anns)} annotations")

    write_split(train_imgs, train_ids, "train")
    write_split(val_imgs,   val_ids,   "val")


# ---------------------------------------------------------------------------
# Step 2 — convert COCO JSON → YOLO .txt labels
# ---------------------------------------------------------------------------

def convert_labels(paths):
    subset_dir = paths["subset_dir"]

    for split in ("train", "val"):
        ann_file = subset_dir / "annotations" / f"instances_{split}.json"
        out_dir  = subset_dir / "labels" / split
        out_dir.mkdir(parents=True, exist_ok=True)

        if out_dir.exists() and any(out_dir.iterdir()):
            print(f"[labels] {split}: already converted, skipping.")
            continue

        print(f"[labels] Converting {split} annotations ...")
        with open(ann_file) as f:
            data = json.load(f)

        images  = {img["id"]: (img["file_name"], img["width"], img["height"])
                   for img in data["images"]}
        cat_ids = sorted(c["id"] for c in data["categories"])
        cat_map  = {cid: i for i, cid in enumerate(cat_ids)}

        ann_by_image: defaultdict = defaultdict(list)
        for ann in data["annotations"]:
            if ann.get("iscrowd", 0) == 1:
                continue
            if "bbox" not in ann or len(ann["bbox"]) != 4:
                continue
            ann_by_image[ann["image_id"]].append(ann)

        for img_id, (fname, w, h) in images.items():
            anns       = ann_by_image[img_id]
            label_path = out_dir / (Path(fname).stem + ".txt")
            if not anns:
                label_path.touch()
                continue
            lines = []
            for ann in anns:
                x, y, bw, bh = ann["bbox"]
                if bw <= 0 or bh <= 0:
                    continue
                cx  = (x + bw / 2) / w
                cy  = (y + bh / 2) / h
                nw  = bw / w
                nh  = bh / h
                cls = cat_map[ann["category_id"]]
                lines.append(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
            label_path.write_text("\n".join(lines))

        n_ann = sum(len(v) for v in ann_by_image.values())
        print(f"[labels] {split}: {len(images)} images, {n_ann} annotations → {out_dir}")


# ---------------------------------------------------------------------------
# Step 3 — train
# ---------------------------------------------------------------------------

def train(paths, args):
    if args.checkpoint:
        ckpt = Path(args.checkpoint).resolve()
        print(f"\n[train] Loading existing checkpoint: {ckpt}")
        model = YOLO(str(ckpt))
        return model, ckpt

    print(f"\n[train] Starting training ...")
    print(f"[train]   config : {paths['config_yaml']}")
    print(f"[train]   data   : {paths['data_yaml']}")
    print(f"[train]   runs → : {paths['runs_dir'] / paths['run_name']}")

    model  = YOLO("yolov8n.pt")
    result = model.train(
        cfg     = str(paths["config_yaml"]),
        data    = str(paths["data_yaml"]),
        project = str(paths["runs_dir"]),
        name    = paths["run_name"],
        exist_ok= True,
    )

    best_pt = Path(result.save_dir) / "weights" / "best.pt"
    print(f"\n[train] Done. Best checkpoint: {best_pt}")
    return model, best_pt


# ---------------------------------------------------------------------------
# Step 4 — evaluate
# ---------------------------------------------------------------------------

def evaluate(model, paths, best_pt):
    print(f"\n[eval] Running COCO-style validation ...")
    metrics = model.val(
        data     = str(paths["data_yaml"]),
        imgsz    = 640,
        save_json= True,
        verbose  = True,
    )

    results = {
        "mAP50":     round(float(metrics.box.map50), 4),
        "mAP50_95":  round(float(metrics.box.map),   4),
        "precision": round(float(metrics.box.mp),    4),
        "recall":    round(float(metrics.box.mr),    4),
        "per_class": {
            model.names[i]: round(float(v), 4)
            for i, v in enumerate(metrics.box.maps)
        },
    }

    out = paths["out_dir"] / f"{paths['run_name']}_coco_eval.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[eval] Saved → {out}")
    print(json.dumps({k: v for k, v in results.items() if k != "per_class"}, indent=2))

    return results, metrics


# ---------------------------------------------------------------------------
# Step 5 — training curves plot
# ---------------------------------------------------------------------------

def plot_training_curves(best_pt, paths):
    run_dir  = best_pt.parent.parent
    csv_path = run_dir / "results.csv"

    if not csv_path.exists():
        print(f"[plot] results.csv not found at {csv_path} — skipping curves.")
        return

    print("\n[plot] Training curves ...")
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].plot(df["epoch"], df["train/box_loss"], label="train")
    axes[0].plot(df["epoch"], df["val/box_loss"],   label="val")
    axes[0].set_title("Box loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(df["epoch"], df["train/cls_loss"], label="train")
    axes[1].plot(df["epoch"], df["val/cls_loss"],   label="val")
    axes[1].set_title("Class loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    axes[2].plot(df["epoch"], df["metrics/mAP50(B)"],    label="mAP@0.5")
    axes[2].plot(df["epoch"], df["metrics/mAP50-95(B)"], label="mAP@0.5:0.95")
    axes[2].set_title("Validation mAP")
    axes[2].set_xlabel("Epoch")
    axes[2].legend()

    run_name = paths["run_name"]
    plt.suptitle(f"{run_name} — training curves", y=1.02)
    plt.tight_layout()
    out = paths["out_dir"] / "training_curves.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] Saved → {out}")


# ---------------------------------------------------------------------------
# Step 6 — per-class mAP bar chart
# ---------------------------------------------------------------------------

def plot_per_class_map(model, metrics, results, paths):
    print("\n[plot] Per-class mAP ...")
    per_class      = results["per_class"]
    sorted_classes = sorted(per_class.items(), key=lambda x: x[1])
    names, scores  = zip(*sorted_classes)
    mean_map       = results["mAP50_95"]

    fig, ax = plt.subplots(figsize=(8, 14))
    ax.barh(names, scores,
            color=["#d9534f" if s < 0.3 else "#5cb85c" for s in scores])
    ax.axvline(mean_map, linestyle="--", color="grey",
               label=f"mean mAP = {mean_map}")
    ax.set_xlabel("mAP@0.5:0.95")
    ax.set_title(f"{paths['run_name']} per-class mAP — worst to best")
    ax.legend()
    plt.tight_layout()
    out = paths["out_dir"] / "per_class_map.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] Saved → {out}")

    print("\n5 worst classes:")
    for name, score in sorted_classes[:5]:
        print(f"  {name:<20} {score:.4f}")


# ---------------------------------------------------------------------------
# Step 7 — sample predictions
# ---------------------------------------------------------------------------

def plot_sample_predictions(model, paths, conf_thresh):
    val_images = paths["val_images"]
    if not val_images.exists():
        print(f"[plot] Val images dir not found: {val_images} — skipping predictions.")
        return

    print(f"\n[plot] Sample predictions (conf >= {conf_thresh}) ...")
    val_imgs = sorted(val_images.glob("*.jpg"))
    sample   = random.sample(val_imgs, min(6, len(val_imgs)))

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for ax, img_path in zip(axes, sample):
        pred = model.predict(str(img_path), conf=conf_thresh, verbose=False)[0]
        img  = Image.open(img_path).convert("RGB")
        ax.imshow(img)
        ax.axis("off")
        for box in pred.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            c   = float(box.conf[0])
            cls = int(box.cls[0])
            lbl = f"{model.names[cls]} {c:.2f}"
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=1.5, edgecolor="#00bfff", facecolor="none",
            )
            ax.add_patch(rect)
            ax.text(x1, y1 - 4, lbl, fontsize=7, color="#00bfff",
                    bbox=dict(facecolor="black", alpha=0.4, pad=1, edgecolor="none"))
        ax.set_title(img_path.name, fontsize=8)

    plt.suptitle(f"{paths['run_name']} predictions (conf >= {conf_thresh})", y=1.01)
    plt.tight_layout()
    out = paths["out_dir"] / "sample_predictions.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] Saved → {out}")


# ---------------------------------------------------------------------------
# Step 8 — confidence distribution
# ---------------------------------------------------------------------------

def plot_conf_distribution(model, paths, n_sample):
    val_images = paths["val_images"]
    if not val_images.exists():
        print(f"[plot] Val images dir not found — skipping confidence distribution.")
        return

    print(f"\n[plot] Confidence distribution ({n_sample} val images) ...")
    val_imgs     = sorted(val_images.glob("*.jpg"))
    sample_paths = random.sample(val_imgs, min(n_sample, len(val_imgs)))
    preds        = model.predict(sample_paths, conf=0.01, verbose=False)
    all_confs    = [c for r in preds for c in r.boxes.conf.tolist()]

    if not all_confs:
        print("[plot] No boxes found — skipping confidence distribution.")
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(all_confs, bins=50, color="steelblue", edgecolor="white")
    ax.axvline(0.25, color="red",    linestyle="--", label="default conf=0.25")
    ax.axvline(0.5,  color="orange", linestyle="--", label="conf=0.5")
    ax.set_xlabel("Confidence score")
    ax.set_ylabel("Count")
    ax.set_title(f"{paths['run_name']} confidence distribution ({n_sample} val images)")
    ax.legend()
    plt.tight_layout()
    out = paths["out_dir"] / "conf_distribution.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] Saved → {out}")

    above = sum(c > 0.5 for c in all_confs)
    print(f"  Total boxes    : {len(all_confs)}")
    print(f"  Median conf    : {np.median(all_confs):.3f}")
    print(f"  Above 0.5      : {above} ({100 * above / len(all_confs):.1f}%)")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(results, paths, best_pt):
    run_name = paths["run_name"]
    out_dir  = paths["out_dir"]

    print()
    print("=" * 50)
    print(f"  {run_name.upper()}")
    print("=" * 50)
    for k, v in results.items():
        if k == "per_class":
            continue
        print(f"  {k:<15} {v:.4f}")
    print()
    print(f"  Checkpoint : {best_pt}")
    print(f"  Results    : {out_dir}")
    print()
    print("  Plots saved:")
    for p in sorted(out_dir.glob("*.png")):
        print(f"    {p.name}")
    print("=" * 50)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args  = parse_args()
    paths = resolve_paths(args)

    if args.checkpoint:
        args.skip_train = True

    # Env info
    print("=" * 50)
    print(f"PyTorch  : {torch.__version__}")
    print(f"CUDA     : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU      : {torch.cuda.get_device_name(0)}")
    print(f"Run      : {paths['run_name']}")
    print(f"Config   : {paths['config_yaml']}")
    print(f"Data     : {paths['data_yaml']}")
    print(f"Out      : {paths['out_dir']}")
    print("=" * 50)

    # Step 1 — subset
    if args.skip_subset:
        print("\n[subset] Skipping (--skip-subset)")
    else:
        train_ok = (paths["subset_dir"] / "images" / "train").exists() and \
                   any((paths["subset_dir"] / "images" / "train").iterdir())
        val_ok   = (paths["subset_dir"] / "images" / "val").exists() and \
                   any((paths["subset_dir"] / "images" / "val").iterdir())
        if train_ok and val_ok:
            print("\n[subset] Images already exist, skipping.")
        else:
            build_subset(paths, args.n_subset, args.train_split, args.seed)

    # Step 2 — labels
    if args.skip_labels:
        print("[labels] Skipping (--skip-labels)")
    else:
        convert_labels(paths)

    # Step 3 — train
    model, best_pt = train(paths, args)

    # Steps 4–8 — eval + plots
    results, metrics = evaluate(model, paths, best_pt)
    plot_training_curves(best_pt, paths)
    plot_per_class_map(model, metrics, results, paths)
    plot_sample_predictions(model, paths, args.conf)
    plot_conf_distribution(model, paths, args.n_sample)
    print_summary(results, paths, best_pt)


if __name__ == "__main__":
    main()
