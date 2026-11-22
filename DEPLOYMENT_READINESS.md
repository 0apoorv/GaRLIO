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

## Timing evidence so far

One recorded full-sequence Python run took about **0.168 seconds per corrected
LiDAR scan on the computer used for that run**. The MATLAB replay uses this as
an assumed service time. At a 10 Hz LiDAR rate, scans arrive every 0.1 seconds,
so a serial processor at that measured average would miss the nominal cadence
and accumulate work. This is an indicative result from one offline run, not a
target-hardware benchmark: it has no per-scan latency distribution, worst-case
latency, queue measurements, or live sensor I/O. See `matlab/README.md` for the
replay limitation.

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
