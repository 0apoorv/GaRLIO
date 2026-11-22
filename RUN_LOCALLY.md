# GaRLIO: local evaluation and MATLAB workflow

This repository contains a Python research pipeline and MATLAB-compatible
trajectory evaluation. The estimator itself runs in Python; the MATLAB folder
recomputes metrics from a `.mat` export and does not port/run the estimator.

## 1. Setup
```
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 2. Run the unit tests
```
PYTHONPATH=src python -m pytest tests/ -v
```

## 3. Extract your bag file (if not already done)
```
python scripts/extract_snail_radar.py /path/to/data1.bag real_data/extracted_data --max_seconds 15
```
This creates real_data/extracted_data/{imu.npy, radar.npz, lidar.npz}
The default is a 15-second extract. To process the entire bag, pass a
`--max_seconds` value longer than the bag duration. Keep the dataset local: it
is not part of this Git repository.

## 4. Put ground truth in place
Copy ref_tls_T_xt32.csv and deviations.csv (from the Snail-Radar ref_trajs download,
sequence 20230920/data1) into:
```
real_data/ground_truth/
```

## 5. Run the demos and full-sequence evaluation
```
PYTHONPATH=src python scripts/demo_propagation.py
PYTHONPATH=src python scripts/demo_dynamic_removal.py
PYTHONPATH=src python scripts/demo_full_pipeline.py
PYTHONPATH=src python scripts/demo_real_data_pipeline.py
PYTHONPATH=src python scripts/evaluate_against_ground_truth.py \
  --data-root /path/to/dataset-root \
  --output-dir outputs/full_sequence
```
`dataset-root` must contain `real_data/extracted_data/` and
`real_data/ground_truth/ref_tls_T_xt32.csv`. The evaluator runs IMU-only and
LiDAR/radar-corrected trajectories, converts estimated IMU poses into the
XT32 LiDAR frame, timestamp-matches ground truth, computes ATE/APE and 1-second
RTE/RRE, and writes plots plus `trajectory_evaluation.mat` under the output
directory.

## 6. Replay a ROS bag through the incremental Python pipeline
The replay reads IMU, radar, and LiDAR messages from the bag in timestamp
order, processes each LiDAR scan through the same incremental estimator used by
the batch runner, and writes per-scan timing and pose outputs:
```sh
PYTHONPATH=src:scripts python scripts/replay_rosbag_realtime.py \
  /path/to/data1.bag --output-dir outputs/stream_replay
```
By default, radar must be within 0.06 seconds of the LiDAR timestamp; override
with `--max-radar-dt` if the dataset's timing requires a different policy. The
script reports measured scan service times and a single-worker deadline/queue
simulation based on recorded timestamps. It reads the bag offline; it does not
connect to live sensors or sleep at the recorded playback rate. Outputs are
`stream_replay_timing.csv`, `stream_replay_trajectory.npz`, and
`stream_replay_timing.png`.
On the local full-sequence run, all 629 poses matched the existing batch
trajectory exactly. After batching the local plane queries, processing averaged
0.1405 s per scan (p95 0.5267 s) and the timestamp schedule missed 599 of 628
inter-scan deadlines. See [DEPLOYMENT_READINESS.md](DEPLOYMENT_READINESS.md)
for the full timing details and limits.

## 7. Run the radar/gravity ablation and evo cross-check
```sh
PYTHONPATH=src python scripts/evaluate_gravity_update_ablation.py \
  --data-root /path/to/dataset-root \
  --output-dir outputs/velocity_gravity_ablation
python scripts/compare_metrics_with_evo.py \
  outputs/full_sequence/trajectory_evaluation.mat
```
The ablation writes a CSV and MAT file for the four radar-Doppler/gravity
update combinations. `evo` is an independent cross-check for APE/ATE and
frame-based RPE; our fixed-time 1-second RTE/RRE are reported separately.

## 8. Recompute metrics in MATLAB or Octave
The MAT export is the hand-off between Python and MATLAB. In MATLAB, from the
repository root:
```matlab
addpath('matlab');
run_evaluation_comparison('outputs/full_sequence/trajectory_evaluation.mat');
```
Or with Octave:
```sh
octave --quiet --eval "addpath('matlab'); run_evaluation_comparison('outputs/full_sequence/trajectory_evaluation.mat');"
```
The script compares ATE/RTE/RRE and pair counts with the Python values saved in
the MAT file. Report a run as MATLAB or Octave validation according to the
application actually used. See [matlab/README.md](matlab/README.md) for each
MATLAB file's role and input conventions.

## 9. Replay recorded poses in MATLAB (real-time-style demonstration)
In MATLAB Online, upload the `matlab/` folder and
`outputs/full_sequence/trajectory_evaluation.mat` to MATLAB Drive. In the
MATLAB command window, run:
```matlab
addpath('matlab');
report = run_realtime_replay( ...
    'outputs/full_sequence/trajectory_evaluation.mat', 0.1405, 1.0);
```
This animates saved corrected poses using their recorded timestamps. `1.0`
plays at the recorded rate; `2.0` is twice as fast. The `0.1405` seconds per
scan is an assumed processing-time input from one optimized offline Python
replay. MATLAB
does not execute the estimator here. Deadline misses and queue wait are
calculated from the assumption, not measured in MATLAB. See
[matlab/README.md](matlab/README.md) before presenting these results.

## Remaining real-time validation
- See [DEPLOYMENT_READINESS.md](DEPLOYMENT_READINESS.md) for the code-based
  deployment assessment, limitations, and concrete checks needed before making
  a live real-time claim.
- Replay another complete sequence and confirm the result is consistent.
- Diagnose why radar Doppler improves translation but slightly increases
  rotation error; verify timestamps, sensor calibration, and frame conventions.
- Evaluate adaptive/fixed covariance changes and compare them against the
  current filter; do not claim false confidence until reported covariance has
  been compared with actual error.
- Port the estimator or connect MATLAB to live Python/ROS before claiming a
  MATLAB real-time estimator test.
- On target hardware, measure per-scan runtime distribution, deadline misses,
  queue growth, dropped/late messages, synchronization, and sensor-loss
  behavior.

## Main files
- `scripts/extract_snail_radar.py`: extracts IMU, radar and LiDAR messages from
  a ROS bag.
- `scripts/demo_real_data_pipeline.py`: Python estimator pipeline and sensor
  update toggles.
- `scripts/evaluate_against_ground_truth.py`: full trajectory evaluation and
  MATLAB MAT export.
- `scripts/evaluate_gravity_update_ablation.py`: radar/gravity four-run
  ablation.
- `scripts/compare_metrics_with_evo.py`: independent evo metric comparison.
- `src/garlio/evaluation.py`: Python timestamp matching, Umeyama alignment,
  ATE and relative-error calculations.
- `src/garlio/snail_radar_extrinsics.py`: dataset sensor extrinsics and pose
  conversion into the XT32 evaluation frame.
- `matlab/`: MATLAB-compatible ATE/RTE/RRE, Python parity, and saved-trajectory
  real-time-style replay scripts.

Generated outputs and the dataset are intentionally kept out of Git; rerun the
commands above to reproduce them locally.
