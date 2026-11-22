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
or configure the MathWorks MATLAB extension in VS Code. MATLAB Online can also
run the `.m` files after you upload the MATLAB folder and MAT export. Octave can
run compatible scripts, but report its results as Octave validation.

## 2. Export the same real-data trajectories from Python

From the repository root, run:

```sh
PYTHONPATH=src:scripts python3.11 scripts/evaluate_against_ground_truth.py \
  --data-root "/path/to/garlio_btp_full" \
  --output-dir outputs/full_sequence
```

The script runs the Python reference evaluation and writes
`outputs/full_sequence/trajectory_evaluation.mat` along with the trajectory and relative-error
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
python scripts/compare_metrics_with_evo.py outputs/full_sequence/trajectory_evaluation.mat
```

The paper does not report RPE. Consecutive-frame RPE is supplementary. The
existing one-second RTE/RRE are also supplementary and remain our own fixed-time
metrics because evo's RPE interface uses frame, distance, or angle deltas rather
than a time delta.

## 4. Run the MATLAB-compatible comparison

With MATLAB, from the repository root:

```sh
matlab -batch "addpath('matlab'); run_evaluation_comparison('outputs/full_sequence/trajectory_evaluation.mat');"
```

With Octave on the checked machine:

```sh
octave --quiet --eval "addpath('matlab'); run_evaluation_comparison('outputs/full_sequence/trajectory_evaluation.mat');"
```

The comparison prints Python and MATLAB-compatible ATE/RTE/RRE RMSE values,
checks matched-pose and interval-pair counts, and errors if any metric differs
by more than `1e-8`. An Octave pass should be reported as Octave validation,
not MATLAB validation.

## 5. Replay the saved trajectory as a real-time-style MATLAB demo
This paces the **saved corrected poses** at their recorded scan timestamps and
animates the path. It also models a serial per-scan processing budget. It does
not run the GaRLIO estimator in MATLAB or read live sensors. The default service
time (0.168 s/scan) is from one offline Python run on this computer; it is not
MATLAB timing or a target-hardware result.

In MATLAB Online, upload the `matlab/` folder and
`outputs/full_sequence/trajectory_evaluation.mat` to MATLAB Drive. In MATLAB,
run:
```matlab
addpath('matlab');
report = run_realtime_replay( ...
    'outputs/full_sequence/trajectory_evaluation.mat', 0.168, 1.0);
```
The same command works in desktop MATLAB from the repository root. Playback
speed `1.0` follows the recorded timing; `2.0` plays twice as fast. The
processing time is an input assumption. The reported queue wait and missed
deadlines are simulated from that assumption, not measured estimator runtime.
Change `0.168` to examine a hypothetical processing budget (for example,
`0.05` s/scan).

For true real-time estimator testing, first port the estimator or connect MATLAB
to the live Python/ROS process. Then measure per-scan runtime and deadline
misses, queue growth, dropped/late messages, timestamp synchronization, and
sensor-loss behavior on the target computer.

## Files

- `align_umeyama.m`: rigid position alignment used by ATE.
- `compute_ate.m`: nearest-time position matching, alignment, and ATE summary.
- `compute_relative_errors.m`: pose matching and fixed-time RTE/RRE.
- `run_evaluation_comparison.m`: load the MAT export and compare with Python.
- `run_realtime_replay.m`: timestamp-paced saved-pose visualization and
  assumption-based serial deadline simulation; not a MATLAB estimator.
