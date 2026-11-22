# MATLAB-based trajectory evaluation

This folder mirrors GaRLIO's Python ATE, 1-second RTE, and 1-second RRE
calculations. It does not port the estimator. The exported estimated poses are
converted into the XT32 frame to match `ref_tls_T_xt32.csv`. Positions use
metres and are stored as `3 x N` columns; rotations are `3 x 3 x N` XT32-to-world
matrices.

## 1. Check the MATLAB installation

In the VS Code integrated terminal on macOS, run:

```sh
command -v matlab
ls -d /Applications/MATLAB*.app
```

If MATLAB is installed but not on `PATH`, launch it from its application folder
or configure the MathWorks MATLAB extension in VS Code to use that installation.
On the checked machine, neither `matlab` nor a MATLAB app was found. GNU Octave
is installed and can run the compatible code below, but that does not count as a
MATLAB run.

## 2. Export the same real-data trajectories from Python

From the repository root, run:

```sh
PYTHONPATH=src:scripts python3.11 scripts/evaluate_against_ground_truth.py \
  --data-root "/Users/apoorvsharma/Downloads/garlio_btp" \
  --output-dir outputs
```

The script runs the Python reference evaluation and writes
`outputs/trajectory_evaluation.mat` along with the trajectory and relative-error
plots. The MAT file contains both estimated trajectories, GT poses, timing
tolerances, and the Python summary values used for parity checks. GT quaternions
are converted from the CSV's `x,y,z,w` columns into rotation matrices before
export. The Python pipeline uses the 20230920/data1 transforms from
`tf_static.launch` for XT32 LiDAR and Oculii radar measurements, and converts
the X36D IMU state poses to XT32 before evaluation. The transform constants are
in `src/garlio/snail_radar_extrinsics.py`.

## 3. Cross-check standard metrics with evo

The paper reports ATE RMSE using evo. The repository's ATE implementation is
kept for MATLAB parity, and this independent script checks its aligned
translation APE/ATE against evo and reports consecutive-frame RPE:

```sh
pip install -r requirements.txt
python scripts/compare_metrics_with_evo.py outputs/trajectory_evaluation.mat
```

The paper does not report RPE. Consecutive-frame RPE is supplementary. The
existing one-second RTE/RRE are also supplementary and remain our own fixed-time
metrics because evo's RPE interface uses frame, distance, or angle deltas rather
than a time delta.

## 4. Run the MATLAB-compatible comparison

With MATLAB, from the repository root:

```sh
matlab -batch "addpath('matlab'); run_evaluation_comparison('outputs/trajectory_evaluation.mat');"
```

With Octave on the checked machine:

```sh
octave --quiet --eval "addpath('matlab'); run_evaluation_comparison('outputs/trajectory_evaluation.mat');"
```

The comparison prints Python and MATLAB-compatible ATE/RTE/RRE RMSE values,
checks matched-pose and interval-pair counts, and errors if any metric differs
by more than `1e-8`. An Octave pass should be reported as Octave validation,
not MATLAB validation.

## Files

- `align_umeyama.m`: rigid position alignment used by ATE.
- `compute_ate.m`: nearest-time position matching, alignment, and ATE summary.
- `compute_relative_errors.m`: pose matching and fixed-time RTE/RRE.
- `run_evaluation_comparison.m`: load the MAT export and compare with Python.
