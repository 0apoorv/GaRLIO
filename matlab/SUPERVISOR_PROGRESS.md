# GaRLIO MATLAB evaluation: progress for review

## Work completed

- Added MATLAB-compatible implementations for ATE, fixed 1-second RTE, and
  fixed 1-second RRE, plus a comparison script that checks the metrics and
  matching counts against the Python values saved in a `.mat` file.
- Checked the evaluation on the Snail-Radar `20230920/data1` sequence. The
  reference file contains PandarXT-32 poses; the estimated X36D IMU poses are
  transformed to the XT32 frame using the sequence's `tf_static.launch`
  calibration before comparison.
- Added the calibrated LiDAR and Oculii radar extrinsics to the Python pipeline.
  The Python test suite passed after these changes (62 tests).
- A previous trajectory export matched Python in MATLAB Online. The latest
  export, after adding velocity-quality weighting and the gravity update, still
  needs a MATLAB Online parity run. Local Octave is currently unable to start
  because of a missing ICU library.
- Added an `evo` cross-check because the paper reports ATE RMSE using `evo`.
  For the current matched poses, evo's translation APE/ATE RMSE matches our
  Umeyama-aligned ATE RMSE to the printed precision. Frame-to-frame RPE is also
  reported as a supplementary metric; the paper itself does not report RPE.

## Preliminary real-data result

This run uses only the first 15 seconds of the sequence, while the ground-truth
file spans about 61.9 seconds. It matched 102 poses, formed 101 consecutive
frame pairs for frame-to-frame RPE, and formed 92 one-second relative intervals.
These are preliminary subset results rather than full-sequence results.

| Metric (RMSE) | Dead reckoning | LiDAR/radar corrected |
|---|---:|---:|
| Translation APE / ATE | 19.058 m | 1.681 m |
| Frame-to-frame RPE translation | 0.910 m | 0.141 m |
| Frame-to-frame RPE rotation | 0.180° | 0.422° |
| 1-second RTE | 9.011 m | 1.372 m |
| 1-second RRE | 0.247° | 1.746° |

The corrected trajectory improves translation APE/ATE, frame-to-frame RPE
translation, and one-second RTE on this subset. Its rotation errors are higher
than dead reckoning for both frame-to-frame RPE and one-second RRE. These
metrics do not establish full-sequence performance.

## Proposed next steps

1. Run the updated `.mat` comparison in MATLAB Online or desktop MATLAB to
   complete the direct MATLAB/Python parity check.
2. Evaluate the full sequence and inspect why corrected RRE is higher.
3. Review the metric definitions, timestamp tolerance, and coordinate-frame
   assumptions with the supervisor.
4. Decide with the supervisor which small estimator component, if any, should
   be ported to MATLAB next.

## Review requested

Are these validation steps sufficient before beginning a MATLAB estimator
port? Which estimator component would be most useful to implement first?
