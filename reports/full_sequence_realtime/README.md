# Full-sequence recorded-data replay

This report contains outputs from a wall-clock-paced replay of the complete Snail-Radar `data1` sequence through the current Python GaRLIO estimator.

- Sensor events: 27,956 IMU, 942 radar, and 629 LiDAR.
- Event span: 67.331 s; playback speed: 1x.
- Playback elapsed: 88.471 s.
- Scan service time: mean 0.1296 s, p95 0.4642 s, maximum 4.6987 s.
- Next-scan deadlines missed: 622 of 628.
- Maximum input queue: 11,791 events.
- Radar measurements omitted at the 0.06 s tolerance: 0.
- Batch comparison: all 629 timestamps matched; maximum position difference 0.000138 m; maximum rotation-matrix element difference 6.246e-6.

The full-sequence event schedule was built locally using the MATLAB generator's ascending time/type/index ordering. That ordering was checked element by element against the MATLAB-generated event manifest on the 15-second sample. The full point-cloud source MAT was not uploaded to MATLAB Drive, so this full-sequence manifest was MATLAB-compatible, not generated in MATLAB Online.

This is a recorded-data simulation on the development Mac. It is not a live sensor, ROS, target-hardware, or direct MATLAB-to-Python test. The missed deadlines show the current implementation did not keep pace in this run.

See [`matlab_realtime_timing.png`](matlab_realtime_timing.png) for processing time, queue wait, and output latency. Per-scan measurements are in [`matlab_realtime_timing.csv`](matlab_realtime_timing.csv); the output poses are in [`matlab_realtime_trajectory.npz`](matlab_realtime_trajectory.npz).
