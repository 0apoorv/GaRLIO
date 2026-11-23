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
time (0.1405 s/scan) is the mean from one optimized offline Python bag replay
on this computer; it is not MATLAB timing or a target-hardware result.

In MATLAB Online, upload the `matlab/` folder and
`outputs/full_sequence/trajectory_evaluation.mat` to MATLAB Drive. In MATLAB,
run:
```matlab
addpath('matlab');
report = run_realtime_replay( ...
    'outputs/full_sequence/trajectory_evaluation.mat', 0.1405, 1.0);
```
The same command works in desktop MATLAB from the repository root. Playback
speed `1.0` follows the recorded timing; `2.0` plays twice as fast. The
processing time is an input assumption. The reported queue wait and missed
deadlines are simulated from that assumption, not measured estimator runtime.
Change `0.1405` to examine a hypothetical processing budget (for example,
`0.05` s/scan).

For true real-time estimator testing, first port the estimator or connect MATLAB
to the live Python/ROS process. Then measure per-scan runtime and deadline
misses, queue growth, dropped/late messages, timestamp synchronization, and
sensor-loss behavior on the target computer.

## 6. Generate a timestamped sensor-event stream for the Python estimator
MATLAB Online's Python environment currently has no NumPy/SciPy, so it cannot
run this estimator directly. MATLAB creates a timestamp-ordered event manifest
from the recorded Snail-Radar sensor messages. The companion source MAT holds
the raw IMU values and radar/LiDAR frames. A local Python producer then emits
those events at their recorded wall-clock rate, while a separate consumer runs
the existing GaRLIO pipeline. This is a file-based real-time replay simulation,
not a direct live MATLAB-to-Python connection or a live sensor test.

First, on the Mac Terminal from the repository root, export the first 15
seconds of raw sensors (about 100 MB):

```sh
PYTHONPATH=src python scripts/export_sensor_stream_for_matlab.py \
  --data-dir "/Users/apoorvsharma/Downloads/garlio_btp_full/real_data/extracted_data" \
  --output outputs/matlab_sensor_stream/sensor_stream_source.mat \
  --max-seconds 15
```

Upload `outputs/matlab_sensor_stream/sensor_stream_source.mat` and
`matlab/generate_sensor_event_stream.m` to MATLAB Drive. In MATLAB Online, run
this in the Command Window:

```matlab
eventScript = dir(fullfile(pwd, '**', 'generate_sensor_event_stream.m'));
sourceFile = dir(fullfile(pwd, '**', 'sensor_stream_source.mat'));
addpath(eventScript(1).folder);
eventFile = fullfile(sourceFile(1).folder, 'matlab_generated_event_stream.mat');
generate_sensor_event_stream( ...
    fullfile(sourceFile(1).folder, sourceFile(1).name), eventFile);
```

Download `matlab_generated_event_stream.mat` from MATLAB Drive into
`outputs/matlab_sensor_stream/`. For the fast offline parity replay, run:

```sh
PYTHONPATH=src:scripts python scripts/replay_matlab_sensor_event_stream.py \
  outputs/matlab_sensor_stream/sensor_stream_source.mat \
  outputs/matlab_sensor_stream/matlab_generated_event_stream.mat \
  --output-dir outputs/matlab_sensor_stream/run
```

It processes the scans without waiting in real time and estimates deadlines
from recorded timestamps. This is useful for quick algorithm parity checks.
For the wall-clock-paced test requested for the MATLAB scenario, run instead:

```sh
PYTHONPATH=src:scripts python scripts/replay_matlab_sensor_event_stream_realtime.py \
  outputs/matlab_sensor_stream/sensor_stream_source.mat \
  outputs/matlab_sensor_stream/matlab_generated_event_stream.mat \
  --output-dir outputs/matlab_sensor_stream/realtime_run \
  --playback-speed 1.0
```

At `1.0x`, an independent producer thread releases events according to the
recorded timestamps while the Python estimator processes incoming events. The
run measures scan service time, event-queue wait, output latency, and missed
next-scan deadlines; it also checks poses against the saved batch trajectory.
The producer and estimator run on the Mac. MATLAB Online supplies the event
manifest; it cannot stream directly into the Mac's local process. Event codes
are IMU=1, radar=2, LiDAR=3; ties are ordered IMU, radar, then LiDAR. Both
scripts use the source MAT payloads by event index.

The Snail-Radar timestamps in the exported 15-second segment have median
intervals of about 2.49 ms for IMU, 66.4 ms for radar, and 100 ms for LiDAR.
These are measured from this sequence, not synthetic frequencies. The paper
describes radar sweeps at a higher rate than LiDAR and radar-based LiDAR sweep
synchronization. Our current pipeline uses the real timestamp order and a
nearest-radar tolerance; it does not implement the paper's full sweep
reconstruction or radar-guided LiDAR dynamic-point removal. Describe this as a
real-time-style test of our current Python implementation, not a complete paper
reproduction.

For the complete local sequence, export through the last LiDAR scan and build
the compact manifest without uploading the large point-cloud MAT to MATLAB
Drive:

```sh
PYTHONPATH=src python scripts/export_sensor_stream_for_matlab.py \
  --data-dir "/Users/apoorvsharma/Downloads/garlio_btp_full/real_data/extracted_data" \
  --output outputs/matlab_sensor_stream/full_sequence_sensor_source.mat \
  --max-seconds 0
python scripts/build_matlab_compatible_event_manifest.py \
  outputs/matlab_sensor_stream/full_sequence_sensor_source.mat \
  outputs/matlab_sensor_stream/full_sequence_event_stream_matlab_compatible.mat
PYTHONPATH=src:scripts python scripts/replay_matlab_sensor_event_stream_realtime.py \
  outputs/matlab_sensor_stream/full_sequence_sensor_source.mat \
  outputs/matlab_sensor_stream/full_sequence_event_stream_matlab_compatible.mat \
  --output-dir outputs/matlab_sensor_stream/full_sequence_realtime_run \
  --playback-speed 1.0
```

The local manifest builder uses the same ordering keys as the MATLAB
`sortrows` generator. We checked all three arrays against MATLAB's downloaded
manifest on the 15-second sample. Label the full run as MATLAB-compatible event
ordering; only the 15-second manifest was generated in MATLAB Online.

## Files

- `align_umeyama.m`: rigid position alignment used by ATE.
- `compute_ate.m`: nearest-time position matching, alignment, and ATE summary.
- `compute_relative_errors.m`: pose matching and fixed-time RTE/RRE.
- `run_evaluation_comparison.m`: load the MAT export and compare with Python.
- `run_realtime_replay.m`: timestamp-paced saved-pose visualization and
  assumption-based serial deadline simulation; not a MATLAB estimator.
- `generate_sensor_event_stream.m`: MATLAB event-order generator for raw
  IMU/radar/LiDAR measurements; consumed by the Python estimator replay.
- `scripts/replay_matlab_sensor_event_stream_realtime.py`: wall-clock-paced
  producer/consumer replay that measures the existing Python estimator.
- `scripts/build_matlab_compatible_event_manifest.py`: local large-sequence
  event scheduler equivalent to the MATLAB `sortrows` ordering.
