# Note to reviewers

This repository releases the source code corresponding to the core method in the manuscript:

1. Continuous Gaussian primitive-field rendering.
2. Conditional diffusion in Gaussian parameter space.
3. Shock-aware Gaussian topology reorganization.
4. Closed-form compressible-flow physical regularization.
5. Multi-resolution inference and evaluation.

The full CFD dataset may contain solver-generated meshes, generated airfoil variants, or institution-specific data. Therefore, the repository includes:

- the exact expected data format,
- the full training/inference/evaluation pipeline,
- a synthetic demo dataset generator,
- and scripts for Gaussian prefitting when CFD fields are available.

To reproduce the main paper numbers, replace the demo data with the CFD samples described in the manuscript and use `configs/pgausdiff.yaml`.
