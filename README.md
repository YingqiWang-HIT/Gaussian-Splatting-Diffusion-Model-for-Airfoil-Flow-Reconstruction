# Physically Constrained Diffusion of Continuous Gaussian Fields

This repository provides a reference implementation of the paper "Physics-Constrained Continuous Reconstruction of Shock-Dominated Transonic Airfoil Flow Fields across Multiple Resolutions".

This method reconstructs transonic airfoil flow fields as continuous anisotropic Gaussian primitive fields. The code contains:

- **CGF-CDG**: Continuous Gaussian field conditional diffusion generator.
- **SA-GTR**: Shock-aware Gaussian topology reorganization by directional split-merge refinement.
- **CD-C³**: Closed-form derivative compressible conservation constraint.
- Multi-resolution Gaussian splatting renderer.
- Training, inference, evaluation, and synthetic demo-data scripts.

> The released code is designed to make the method, training pipeline, and evaluation protocol transparent. The CFD dataset used in the paper should be placed under `data/` following the format in `docs/data_format.md`. 

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
├── pyproject.toml
└── requirements.txt
```

## Installation

Please refer to requirements.txt for details.

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

If you find this repository useful, please cite our paper:

```bibtex
@article{wang2026physics,
  title   = {Physics-Constrained Continuous Reconstruction of Shock-Dominated Transonic Airfoil Flow Fields across Multiple Resolutions},
  author  = {Wang, Yingqi and Song, Yuchen and Wang, Wentao and Zhang, Yusu and Liu, Datong},
  journal = {Aerospace Science and Technology},
  year    = {2026},
  pages   = {113044},
  issn    = {1270-9638},
  doi     = {10.1016/j.ast.2026.113044},
  url     = {https://www.sciencedirect.com/science/article/pii/S1270963826014227}
}
```
