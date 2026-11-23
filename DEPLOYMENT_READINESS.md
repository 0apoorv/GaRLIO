# GaRLIO deployment readiness

## Assessment

**Our current Python implementation is an offline research prototype with a
wall-clock-paced recorded-data replay, not a live sensor estimator.** MATLAB
creates an event schedule from the Snail-Radar recording; a local Python
producer releases events at their recorded timestamps while a separate
consumer runs the existing estimator. MATLAB Online does not connect directly
to the local Python process, and there is no live sensor or ROS subscription.

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
  each LiDAR frame. The MATLAB event-stream replay adds a bounded event queue
  and buffers measurements, but there is no live ROS subscriber or pose
  publisher.
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

## MATLAB-scheduled wall-clock replay milestone

The first 15-second Snail-Radar export contains 7,041 IMU samples, 225 radar
frames, and 150 LiDAR scans. Their timestamp medians are approximately 2.49 ms
(about 402 Hz), 66.4 ms (about 15.1 Hz), and 100 ms (about 10 Hz), respectively.
These rates were calculated from this recording; the paper describes radar as
higher frequency than LiDAR and synchronizes radar/LiDAR sweeps, but does not
specify these exact frequencies for our exported segment.

A wall-clock-paced replay released the 7,416 events over 19.53 seconds while
the estimator processed all 150 LiDAR scans. The downloaded MATLAB event MAT
file was checked against a local reconstruction: `eventTimes`, `eventTypes`,
and `eventIndices` were identical element by element. Across four paced runs
(two used the downloaded MATLAB file), playback took 20.31-21.05 s; scan
service mean was 0.0497-0.0712 s, p95 was 0.0722-0.1616 s, and max was
0.1046-0.4409 s. Output missed 50-91 of 149 next-scan deadlines; p95
pose-output latency from scheduled LiDAR time was 0.1260-0.4835 s. No radar
match was omitted at the configured 0.06 s tolerance. Timestamps, positions,
and rotations matched the saved batch trajectory exactly in all four runs.
The timing spread shows that performance varies substantially between runs on
this development machine; the worst run had a backlog spike and many missed
deadlines.

This shows that the current estimator consumed a wall-clock-paced recorded
schedule on the development Mac, with missed deadlines and variable timing. It
is not a target-hardware or live-sensor result. The actual downloaded MATLAB
event manifest was replayed and its arrays matched the local reconstruction,
verifying the file handoff and event ordering.

## Full-sequence wall-clock replay

The complete Snail-Radar sequence was exported locally through its final
LiDAR scan: 27,956 IMU samples, 942 radar frames, and 629 LiDAR scans across a
67.331 s event span. The full event manifest was built locally using the same
ascending time/type/index ordering as the MATLAB script. This ordering was
verified element by element against the downloaded MATLAB manifest on the
15-second sample; the full-size raw point-cloud MAT was not uploaded to MATLAB
Drive and the full manifest itself was not generated in MATLAB Online.

At 1x playback, all 629 scans were processed. The run took 88.47 s for 67.33 s
of recorded events; service time was mean 0.1296 s, p95 0.4642 s, and max
4.6987 s. It missed 622 of 628 next-scan deadlines, with p50/p95 output
latencies of 20.12/24.98 s and a maximum input-event queue of 11,791. No radar
match was omitted within 0.06 s. The resulting trajectory matched all 629
batch timestamps; maximum position difference was 0.000138 m and maximum
rotation-matrix element difference was 6.25e-6. This confirms trajectory
consistency, while the queue growth and missed deadlines show that the current
Python implementation cannot keep up with this full sequence in the measured
scenario.

This test is not a complete reproduction of GaRLIO. The paper describes
radar/LiDAR sweep reconstruction, radar-guided dynamic-point removal, and
motion compensation. Our runner sorts measurements and uses the nearest radar
frame within a fixed tolerance; those paper-specific preprocessing steps are
outside this test.

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

For the BTP, we can demonstrate **MATLAB-scheduled, wall-clock-paced playback
through our Python estimator** and report measured development-machine timing
and batch parity. We cannot claim direct MATLAB-to-Python streaming, live sensor
input, complete paper reproduction, or deployment readiness. Next, repeat with
the actual downloaded MATLAB manifest, then add a local MATLAB-to-Python
transport or ROS interface and evaluate the paper's remaining preprocessing
before measuring on target hardware.

## Sources

- Local implementation: `scripts/demo_real_data_pipeline.py`,
  `scripts/extract_snail_radar.py`, and `src/garlio/dynamic_removal.py`.
- Local replay notes and assumptions: `matlab/README.md`.
- [Author GaRLIO repository](https://github.com/ChiyunNoh/GaRLIO), including
  ROS1 package/build/launch instructions and stated tested environment.
- [GaRLIO paper, Sections IV and V](https://arxiv.org/html/2502.07703v2),
  describing scan synchronization, preprocessing, updates, and evaluation.
