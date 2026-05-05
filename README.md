# PGaus-Diff: Physics-Constrained Continuous Gaussian Field Diffusion

This repository provides the reference implementation for the manuscript:

**Physics-Constrained Continuous Reconstruction of Shock-Dominated Transonic Airfoil Flow Fields across Multiple Resolutions**

PGaus-Diff reconstructs transonic airfoil flow fields as continuous anisotropic Gaussian primitive fields. The code contains:

- **CGF-CDG**: Continuous Gaussian field conditional diffusion generator.
- **SA-GTR**: Shock-aware Gaussian topology reorganization by directional split-merge refinement.
- **CD-C³**: Closed-form derivative compressible conservation constraint.
- Multi-resolution Gaussian splatting renderer.
- Training, inference, evaluation, and synthetic demo-data scripts.

> The released code is designed to make the method, training pipeline, and evaluation protocol transparent. The CFD dataset used in the paper should be placed under `data/` following the format in `docs/data_format.md`. A small synthetic demo is included only to verify the pipeline.

## Repository structure

```text
PGaus-Diff-GitHub/
├── configs/
│   ├── demo.yaml
│   └── pgausdiff.yaml
├── docs/
│   ├── data_format.md
│   └── reviewer_code_note.md
├── examples/
│   └── run_demo.sh
├── pgausdiff/
│   ├── __init__.py
│   ├── config.py
│   ├── data.py
│   ├── diffusion.py
│   ├── gaussian.py
│   ├── metrics.py
│   ├── model.py
│   ├── physics.py
│   ├── refinement.py
│   └── utils.py
├── scripts/
│   ├── evaluate.py
│   ├── infer.py
│   ├── make_demo_data.py
│   ├── prefit_gaussians.py
│   └── train.py
├── LICENSE
├── CITATION.cff
├── pyproject.toml
└── requirements.txt
```

## Installation

```bash
git clone https://github.com/<your-name>/PGaus-Diff.git
cd PGaus-Diff
conda create -n pgausdiff python=3.12 -y
conda activate pgausdiff
pip install -r requirements.txt
pip install -e .
```

The manuscript experiments used Python 3.12, PyTorch 2.7, CUDA 12.9, two NVIDIA RTX 5090 GPUs, and an Intel Core i9-14900KF CPU. The demo can run on CPU or a single GPU.

## Quick demo

```bash
bash examples/run_demo.sh
```

The demo will:

1. Generate small synthetic airfoil-like flow fields.
2. Fit Gaussian primitive parameters for each field.
3. Train a small conditional diffusion model.
4. Run inference and render a higher-resolution field.
5. Evaluate PSNR, SSIM, and conservation residuals.

## Training on your CFD dataset

Prepare `.npz` files according to `docs/data_format.md`, then run:

```bash
python scripts/prefit_gaussians.py --config configs/pgausdiff.yaml
python scripts/train.py --config configs/pgausdiff.yaml
python scripts/infer.py --config configs/pgausdiff.yaml --checkpoint outputs/checkpoints/best.pt
python scripts/evaluate.py --config configs/pgausdiff.yaml --pred_dir outputs/predictions
```

## Data availability

The code is released for method reproducibility. If the CFD dataset contains proprietary meshes, commercial-solver outputs, or unpublished airfoil geometries, provide either:

- a public subset,
- synthetic/demo samples,
- preprocessing scripts and exact data format,
- or a data-availability statement explaining access restrictions.

## Main hyperparameters used in the paper

| Module | Hyperparameter | Value |
|---|---:|---:|
| Gaussian primitives | Number of primitives | 3000 |
| CGF-CDG | Geometry Conv1D layers | 4 |
| CGF-CDG | Condition MLP layers | 3 |
| CGF-CDG | Embedding dimension | 512 |
| CGF-CDG | Attention heads | 8 |
| CGF-CDG | Transformer layers | 8 |
| SA-GTR | Refinement rounds | 3 |
| SA-GTR | Fine-tuning steps per round | 50 |
| SA-GTR | Split scaling factor | 0.5 |
| SA-GTR | Split percentile | 90 |
| CD-C³ | Collocation points | 10000 |
| CD-C³ | Control volumes | 500 |
| Training | Diffusion steps | 2000 |
| Training | Sampling steps | 100 |
| Training | Batch size | 32 |
| Training | Epochs | 500 |

For the quick demo, these values are reduced in `configs/demo.yaml`.

## Citation

If this code is useful, please cite the corresponding paper. See `CITATION.cff`.
