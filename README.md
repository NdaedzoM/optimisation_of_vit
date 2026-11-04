# Optimizing Vision Transformers for Medical and Natural Image Segmentation

This repository hosts the complete, modular codebase (`code_v2`) designed to evaluate, optimize, and deploy advanced Vision Transformers (ViTs) on medical and natural image segmentation tasks. 

The project focuses on benchmarking architectures, including **Swin-UNet**, **TransUNet**, and **SegFormer**, under a unified training framework, exploring optimizations in efficiency, latency, parameter complexity, and precision across both domains: **BraTS 2020** (multimodal brain tumor glioblastoma MRI scans) and **ADE20K** (natural scenes semantic layout parsing).

---

## Core Directory Layout
```bash
code_v2/
├── configs/                 # YAML parameters configurations
├── models/                  # Network architectures
│   ├── backbones/           # Swin-Transformer, patch embeds, and modules
│   ├── baselines/           # TransUNet (R50-ViT-B/16), SegFormer baseline modules
│   ├── factory.py           # Config-driven model initialization
│   ├── utils.py             # Parameter calculations, channel inflation, and depthwise conv adaptations
│   └── optimizer_engine.py  # H1 (Res-DW-Adapter) and H2 (SATP Spatially Aware Token Pruning)
├── training/                # Pipeline routines
│   ├── train.py             # Entry script for end-to-end training runs
│   ├── trainer.py           # Optimization algorithms, AMP, and step scheduling
│   └── run_queue.py         # Multi-run queue script to execute the priority matrix across seeds
├── evaluation/              # Deployed metrics and benchmark suites
│   ├── metrics.py           # Patient-level 3D volume (Dice, HD95) and class-level IoU accumulators
│   ├── benchmark.py         # H3 Engine deployment (ONNX export, TensorRT engine compilation, FP16/INT8)
│   ├── visualize.py         # Visualizing predictions, SATP masks, and Pareto frontiers
│   └── aggregate.py         # Tabulating metrics, LaTeX formatting, and Wilcoxon signed-rank tests
└── utils/
    ├── dataloader.py        # Dataset streaming pipelines for BraTS2020 and ADE20K
    ├── preprocess.py        # Center-cropping, percent-clipping, z-scoring, and class mapping
    └── losses.py            # Custom soft Dice + Cross Entropy (CE) loss
```

---

## 6E0þ0F Step-by-Step Instructions

### 1. Offline Preprocessing
Datasets are packaged into `.npz` slices to accelerate training bandwidth. This converts 3D medical volumes into structured axial slices and standardizes natural images:
```bash
# Preprocess BraTS2020
python utils/preprocess.py --dataset brats --out /content/brats_slices

# Preprocess ADE20K
python utils/preprocess.py --dataset ade20k --out /content/ade_npz
```

### 2. End-to-End Training
Train architectures using a single-command syntax with dynamic override configurations. Large model metrics are backed up to Drive on every epoch:
```bash
# Train baseline Swin-UNet on BraTS2020
python training/train.py --model swin_unet --dataset brats --seed 42

# Train UOViT-E (H1 + H2) on ADE20K
python training/train.py --model swin_unet --dataset ade20k --adapter --keep-ratio 0.75 --seed 42
```

### 3. Pipeline Automation
Run the complete priority matrix over multiple seeds (`42`, `43`, `44`) sequentially using the queue module:
```bash
python training/run_queue.py --dataset brats --seeds 42 43 44
```

### 4. Deployment Benchmarking (H3)
Export trained PyTorch checkpoints to explicit Q/DQ strongly-typed ONNX graphs, compile to TensorRT engines, and perform INT8 entropy calibration:
```bash
python evaluation/benchmark.py --run "experiments/swin_unet-d2262_brats_h1_h2r0.75_s42" --precisions fp32 fp16 int8
```

### 5. Quantitative Synthesis & Metrics
Collate experimental outputs, calculate statistical Wilcoxon p-values (Holm-corrected), generate LaTeX tables, and plot Pareto frontiers or token maps:
```bash
# Generate Tables
python evaluation/aggregate.py

# Generate Visualizations
python evaluation/visualize.py --pareto
python evaluation/visualize.py --paper --runs "experiments/swin_unet-d2262_brats_h1_h2r0.75_s42"
```
