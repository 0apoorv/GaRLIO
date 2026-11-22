# GaRLIO deployment readiness

## Assessment

**Our current Python implementation is an offline research prototype, not a
live real-time estimator.** It reads complete extracted sensor arrays, processes
LiDAR frames in a loop, and saves trajectories for evaluation. The MATLAB work
plays back saved poses; it does not run the estimator or connect to live
sensors.

The authors' public implementation is further along as a robotics package: its
repository contains a ROS1/C++ package, launch files, and instructions to play a
ROS bag. The README lists Ubuntu 20.04, ROS Noetic, PCL 1.10, and Eigen 3.1 as
the tested environment. This gives us a useful deployment reference, but bag
playback instructions and the listed environment do not by themselves prove
that it meets live sensor deadlines on our target hardware.

## What our code currently does

- `scripts/extract_snail_radar.py` reads a recorded ROS bag offline and saves
  sensor streams as NumPy files.
- `scripts/demo_real_data_pipeline.py` consumes those saved arrays and processes
  each LiDAR frame; there is no live ROS subscriber, bounded streaming buffer,
  or pose publisher in this runner.
- The estimator selects the nearest radar frame for each LiDAR timestamp with
  no maximum allowed time difference. It scans the IMU array to select each
  propagation segment.
- `src/garlio/dynamic_removal.py` contains a radar-guided LiDAR dynamic-removal
  helper, but the real-data runner does not call it.
- The paper describes radar/LiDAR sweep synchronization, filling missing scan
  fields of view from the previous sweep, LiDAR motion compensation, and
  radar-guided dynamic LiDAR removal. Those steps are not present in the
  current real-data runner, so our Python pipeline should not be described as a
  complete reproduction of the paper.

## First Python bag-replay milestone

The incremental processor was run against the local Snail-Radar `data1.bag`.
It consumed 28,645 IMU messages, 944 radar frames, and 629 LiDAR scans in bag
timestamp order. No scan lacked radar within the configured 0.06-second
tolerance. The exported 629 poses matched the existing batch trajectory
exactly: timestamps, positions, and rotations had zero difference at the
reported numeric precision.

An initial replay profile found that local KD-tree queries and plane fits
accounted for much of scan-processing time. The implementation now batches the
queries and plane fits for each EKF residual evaluation. A follow-up full-bag
run processed all 629 scans with median LiDAR period **0.1000 s**. Per-scan
processing was mean **0.1405 s**, p50 **0.0580 s**, p95 **0.5267 s**, and max
**2.5125 s**. The single-worker timestamp schedule missed **599 of 628
inter-scan deadlines**; simulated output latency was p50 **24.30 s** and p95
**27.29 s**. Every scan still had a radar frame within 0.06 s.

The optimization preserved the full trajectory exactly: all 629 timestamps,
positions, and rotations match the saved batch trajectory. The average service
time is lower than the 0.1 s scan period, but the long tail means this run
still does not establish reliable 10 Hz operation.

The separate initial replay measured a 0.8382 s mean, so the observed full-run
mean fell by about **83%** after batching; these two full-run measurements were
not collected under a controlled benchmark setup.

As a focused profile check, the same 15 scans (26–40) took 6.305 s total in
`PipelineProcessor.process_scan` before batching and 1.037 s afterward under
the profiler. This supports the optimization result, though the full replay
still needs repeated runs under controlled load.

This is an offline bag read and timestamp-based schedule simulation, not a live
sensor test. It does not measure actual ROS queue growth or target-hardware
latency. Repeat under controlled, documented hardware/load conditions before
making a deployment claim. See `matlab/README.md` for the saved-pose playback
limitation.

## Work needed before claiming real-time deployment

1. **Close the algorithm and data-path gaps.** Compare our implementation
   against the author code and paper; implement or explicitly scope out sweep
   reconstruction, timestamp synchronization, motion compensation, and
   dynamic-point removal. Reject stale radar frames using a configured time
   tolerance and define behavior for missing or delayed sensors.
2. **Add a streaming interface.** Receive timestamped IMU, radar, and LiDAR
   messages, buffer them with bounded memory, process scans in timestamp order,
   and publish poses plus health/status information. The author's ROS1 package
   is a reference starting point; the target ROS version and sensor drivers
   still need to be selected.
3. **Measure and optimize on the intended computer.** Record per-stage and
   per-scan p50/p95/p99/max latency, deadline misses, queue depth, CPU, and
   memory at the sensor's actual rate. Optimize the hot path if it cannot keep
   up; the current pipeline repeatedly constructs/query-fits local planes and
   performs Python-level point loops.
4. **Test operational failures.** Exercise timestamp skew, dropped/out-of-order
   messages, stale radar, sensor outages, startup/restart, and long-duration
   runs. Check that the estimator reports degraded or invalid output rather
   than silently using bad inputs.
5. **Validate accuracy and uncertainty.** Repeat the full-sequence APE/ATE and
   relative-error evaluation after changes, across more sequences. To assess
   possible overconfidence, log filter covariance and compare predicted
   uncertainty with observed pose errors; current trajectory metrics alone do
   not establish this.

## Practical conclusion

For the BTP, we can demonstrate **offline evaluation and timestamp-paced pose
replay** now. We cannot yet claim that our Python estimator is ready for live
deployment or that MATLAB tested the estimator in real time. The next useful
deployment study is to build and replay the authors' ROS package in its
documented environment, compare its outputs with our Python results, and then
measure actual scan-processing deadlines on the intended hardware. If the BTP
scope remains MATLAB-only, present the MATLAB playback as a system-level
simulation and clearly label its processing-time input as assumed or taken
from one offline run.

## Sources

- Local implementation: `scripts/demo_real_data_pipeline.py`,
  `scripts/extract_snail_radar.py`, and `src/garlio/dynamic_removal.py`.
- Local replay notes and assumptions: `matlab/README.md`.
- [Author GaRLIO repository](https://github.com/ChiyunNoh/GaRLIO), including
  ROS1 package/build/launch instructions and stated tested environment.
- [GaRLIO paper, Sections IV and V](https://arxiv.org/html/2502.07703v2),
  describing scan synchronization, preprocessing, updates, and evaluation.
