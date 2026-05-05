# Data format

Each CFD sample is stored as one `.npz` file under:

```text
data/pgausdiff/
├── train/
│   ├── sample_000001.npz
│   └── ...
└── test/
    ├── sample_000001.npz
    └── ...
```

## Required keys

| Key | Shape | Description |
|---|---:|---|
| `geometry` | `[Ns, 2]` | Ordered airfoil surface coordinates, normalized to the computational domain. In the paper, `Ns=200`. |
| `condition` | `[3]` | Incoming condition `[Mach, Reynolds, angle_of_attack_deg]`. |
| `theta` | `[N, 9]` | Gaussian primitive parameters. In the paper, `N=3000`. |

## Gaussian primitive layout

Each row of `theta` is:

```text
[x, y, sx, sy, angle, rho, u, v, p]
```

where:

- `(x, y)` is the primitive center.
- `(sx, sy)` are positive principal-axis scales.
- `angle` is the rotation angle in radians.
- `(rho, u, v, p)` are primitive physical variables.

The rendered continuous field is:

```text
q(x) = sum_i w_i * phi_i(x) / (sum_i phi_i(x) + eps)
```

with `w_i = [rho_i, u_i, v_i, p_i]`.

## Optional keys

| Key | Shape | Description |
|---|---:|---|
| `field` | `[H, W, 4]` | CFD reference field `[rho, u, v, p]` on a grid. |
| `query_xy` | `[H, W, 2]` or `[M, 2]` | Coordinates corresponding to `field`. |
| `wall_xy` | `[Nw, 2]` | Airfoil wall coordinates for boundary-condition loss. |
| `wall_normals` | `[Nw, 2]` | Outward wall normals for impermeability loss. |

## Dataset construction used in the manuscript

The paper used three groups of airfoils:

- NACA0012-CST: 100 airfoils, 20 conditions, six resolution types.
- RAE2822-CST: 100 airfoils, 20 conditions, six resolution types.
- UIUC: 300 airfoils, 20 conditions, six resolution types.

The split is geometry-disjoint with an 8:2 train/test ratio.

## Recommended preprocessing workflow

1. Export CFD fields `[rho, u, v, p]` and query coordinates from the solver.
2. Normalize coordinates to `[0, 1] × [0, 1]` or update the domain in the renderer.
3. Fit Gaussian primitives with `scripts/prefit_gaussians.py`, or provide your own fitted `theta`.
4. Compute global `theta_min` and `theta_max` on the training set, then fill them in `configs/pgausdiff.yaml`.
5. Train PGaus-Diff with `scripts/train.py`.
