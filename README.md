# BeyondTiePoints

BeyondTiePoints is a remote-sensing image alignment framework built around:

1. **Encoder pretraining** for robust, confidence-aware dense features.
2. **PBA (Pyramid Bundle Adjustment)** for multi-level RPC affine refinement over overlapping image grids.

The repository includes both the feature pretraining pipeline (`pretrain/`) and the final adjustment pipeline (`main.py` + `adjustment_core/`).

---

## Table of Contents

- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Environment Setup](#environment-setup)
- [Encoder Pretraining](#encoder-pretraining)
  - [Pretraining Data Format](#pretraining-data-format)
  - [Run Pretraining](#run-pretraining)
  - [Expected Outputs](#expected-outputs)
- [PBA](#pba)
  - [PBA Input Data Format](#pba-input-data-format)
  - [Run PBA](#run-pba)
  - [Key Arguments](#key-arguments)
  - [Outputs](#outputs)
- [Practical Notes](#practical-notes)
- [Citation](#citation)

---

## Introduction

Tie points are often sparse, noisy, or unstable under large appearance/geometry changes in satellite imagery. This project replaces pure tie-point dependence with a feature-driven alignment strategy:

- A pretrained encoder predicts dense descriptors and confidence maps.
- The scene overlap is partitioned into geographic grids.
- For each grid, pairwise feature consistency losses are used to optimize per-image affine corrections on RPC models.
- A pyramid strategy progressively refines corrections from coarse to fine windows.

The implementation supports both single-process execution and distributed execution via the `--use_ddp` switch.

---

## Project Structure

```text
.
├── main.py                       # Entry point for PBA
├── adjustment_core/
│   ├── data.py                   # SharedGrid construction and feature extraction
│   ├── ddp.py                    # Distributed runtime wrapper
│   ├── grid.py                   # Grid generation/selection/subdivision/visualization
│   ├── loop.py                   # Optimization loop and feature sampling
│   ├── model.py                  # Affine parameter model
│   ├── utils.py                  # Orthorectification, checkerboard, loaders, logger
│   └── validation.py             # Error report utilities
├── model/
│   ├── encoder.py                # EncoderDino + adapter/confidence head
│   └── decoder.py                # Decoder used in pretraining
├── pretrain/
│   ├── pretrain.py               # Encoder pretraining entry point
│   ├── dataloader.py             # H5 dataset loader and sampling
│   └── criterion.py              # Pretraining losses
├── rpc.py                        # RPC model + affine update/merge
├── rs_image.py                   # RS image wrapper
└── env.yaml                      # Conda environment spec (Python 3.10)
```

---

## Environment Setup

Use the provided `env.yaml` (Python 3.10):

```bash
conda env create -f env.yaml
conda activate beyond-tie-points
```

> The current environment file targets PyTorch + CUDA 12.1 builds. If you need CPU-only runtime, adjust torch/torchvision/cuda packages accordingly.

---

## Encoder Pretraining

### Pretraining Data Format

Pretraining expects HDF5 datasets under `--dataset_path`:

- `train_data.h5`
- `test_data.h5`

Each sample key contains at least:

- `images/image_i` (grayscale image)
- `obj` (object-space map)
- `residuals/residual_i`

See `pretrain/dataloader.py` for exact indexing and fields.

### Run Pretraining

A typical distributed launch:

```bash
torchrun --nproc_per_node=4 pretrain/pretrain.py \
  --dataset_path ./datasets \
  --dino_weight_path ./weights/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth \
  --encoder_output_path ./weights/pretrain_run \
  --batch_size 8 \
  --max_epoch 200 \
  --lr_encoder_max 5e-4 \
  --lr_decoder_max 1e-3
```

Single-process debugging example:

```bash
python pretrain/pretrain.py \
  --dataset_path ./datasets \
  --dino_weight_path ./weights/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth \
  --encoder_output_path ./weights/pretrain_debug \
  --batch_size 2 \
  --max_epoch 2
```

### Expected Outputs

Depending on your arguments, pretraining writes:

- Encoder/adaptor checkpoints
- Decoder checkpoints
- Logs and optional visualizations

Use the produced `adapter.pth` as `--adapter_path` in PBA.

---

## PBA

### PBA Input Data Format

Set `--root` to a directory containing:

```text
<root>/
└── adjust_images/
    ├── <image_folder_0>/
    │   ├── image.png
    │   ├── dem.npy
    │   ├── rpc.txt
    │   └── tie_points.txt   # optional
    ├── <image_folder_1>/
    │   ├── image.png
    │   ├── dem.npy
    │   └── rpc.txt
    └── ...
```

### Run PBA

```bash
python main.py \
  --root /path/to/project_data \
  --dino_path ./weights \
  --adapter_path ./weights/pretrain_run/adapter.pth \
  --use_ddp auto \
  --num_levels 2 \
  --window_size 2000 \
  --grid_num 16 \
  --max_iter 1000
```

Distributed launch example:

```bash
torchrun --nproc_per_node=4 main.py \
  --root /path/to/project_data \
  --dino_path ./weights \
  --adapter_path ./weights/pretrain_run/adapter.pth \
  --use_ddp true \
  --num_levels 2 \
  --window_size 2000
```

### Key Arguments

- `--adapter_path`: pretrained adapter checkpoint used by `EncoderDino`.
- `--use_ddp {auto,true,false}`:
  - `auto`: enable DDP when `WORLD_SIZE > 1`
  - `true`: force DDP
  - `false`: single-process path
- `--num_levels`: number of pyramid levels.
- `--window_size`: base grid size (meters) for level 0.
- `--grid_num`: number of selected grids in level 0.
- `--max_grid_num`: cap for subdivided grids in deeper levels.
- `--select_grid_by_conf`: enable confidence-based grid selection.
- `--patience`, `--min_loss_threshold`: early stopping controls.

### Outputs

Under `<root>/debug_output` (or `<root>/output_<experiment_id>`):

- Grid visualizations
- Checkerboard visualizations per grid
- Per-level baked RPC files
- `loss_log_level_*.txt`
- `final_results.json`

---

## Practical Notes

1. `EncoderDino` loads DINO via local hub path (`./dinov3` in `model/encoder.py`). Ensure the local DINO repo/weights are prepared before running.
2. For reproducibility, keep `--random_seed` fixed.
3. For large scenes, start with smaller `--grid_num` / fewer levels to validate pipeline health before full-scale runs.
4. If using multi-GPU, always launch with `torchrun` and consistent environment variables.

---

## Citation

If this code helps your work, please cite your paper and/or this repository in your publication.

```bibtex
@misc{beyondtiepoints,
  title  = {BeyondTiePoints},
  author = {Authors},
  year   = {2026},
  note   = {Code repository}
}
```
