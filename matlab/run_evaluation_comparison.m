function results = run_evaluation_comparison(matPath)
%RUN_EVALUATION_COMPARISON Compare MATLAB-compatible metrics to Python values.
% From the repository root:
%   octave --quiet --eval "addpath('matlab'); run_evaluation_comparison('outputs/trajectory_evaluation.mat');"
% The same function can be called from MATLAB after adding this folder to path.

if nargin < 1 || isempty(matPath)
    repoRoot = fileparts(fileparts(mfilename('fullpath')));
    matPath = fullfile(repoRoot, 'outputs', 'trajectory_evaluation.mat');
end
if exist(matPath, 'file') ~= 2
    error('MAT input file not found: %s', matPath);
end
addpath(fileparts(mfilename('fullpath')));
data = load(matPath);

ateDr = compute_ate(data.times_dr, data.positions_dr, ...
    data.gt_times, data.gt_positions, data.timestamp_tolerance_s);
ateCorrected = compute_ate(data.times_corrected, data.positions_corrected, ...
    data.gt_times, data.gt_positions, data.timestamp_tolerance_s);
relDr = compute_relative_errors(data.times_dr, data.positions_dr, ...
    data.rotations_dr, data.gt_times, data.gt_positions, data.gt_rotations, ...
    data.relative_interval_s, data.timestamp_tolerance_s, data.interval_tolerance_s);
relCorrected = compute_relative_errors(data.times_corrected, ...
    data.positions_corrected, data.rotations_corrected, data.gt_times, ...
    data.gt_positions, data.gt_rotations, data.relative_interval_s, ...
    data.timestamp_tolerance_s, data.interval_tolerance_s);

tol = 1e-8;
fprintf('MATLAB-compatible evaluation of %s\n', matPath);
fprintf('Metric                    Python          MATLAB          abs diff\n');
printMetric('ATE RMSE dead-reckoning', data.python_ate_dr_rmse, ateDr.rmse);
printMetric('ATE RMSE corrected', data.python_ate_corrected_rmse, ateCorrected.rmse);
printMetric('RTE RMSE dead-reckoning', data.python_rte_dr_rmse, relDr.translation_rmse);
printMetric('RTE RMSE corrected', data.python_rte_corrected_rmse, relCorrected.translation_rmse);
printMetric('RRE RMSE DR [rad]', data.python_rre_dr_rmse_rad, relDr.rotation_rmse_rad);
printMetric('RRE RMSE corrected [rad]', data.python_rre_corrected_rmse_rad, relCorrected.rotation_rmse_rad);

if ateDr.n_matched ~= data.python_ate_dr_n_matched || ...
        ateCorrected.n_matched ~= data.python_ate_corrected_n_matched || ...
        relDr.n_pairs ~= data.python_relative_dr_n_pairs || ...
        relCorrected.n_pairs ~= data.python_relative_corrected_n_pairs
    error('Python and MATLAB-compatible matching counts differ');
end
differences = [abs(ateDr.rmse - data.python_ate_dr_rmse), ...
    abs(ateCorrected.rmse - data.python_ate_corrected_rmse), ...
    abs(relDr.translation_rmse - data.python_rte_dr_rmse), ...
    abs(relCorrected.translation_rmse - data.python_rte_corrected_rmse), ...
    abs(relDr.rotation_rmse_rad - data.python_rre_dr_rmse_rad), ...
    abs(relCorrected.rotation_rmse_rad - data.python_rre_corrected_rmse_rad)];
if any(differences > tol)
    error('Metric comparison exceeds tolerance %.1e', tol);
end
fprintf('Counts match; all metric differences are <= %.1e.\n', tol);

results.ate_dead_reckoning = ateDr;
results.ate_corrected = ateCorrected;
results.relative_dead_reckoning = relDr;
results.relative_corrected = relCorrected;
results.max_abs_metric_difference = max(differences);
results.tolerance = tol;
end

function printMetric(name, pythonValue, matlabValue)
fprintf('%-27s %12.9f  %12.9f  %.3g\n', ...
    name, pythonValue, matlabValue, abs(pythonValue - matlabValue));
end
