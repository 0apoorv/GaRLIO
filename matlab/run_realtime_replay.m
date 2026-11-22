function report = run_realtime_replay(matPath, processingTimeS, playbackSpeed)
%RUN_REALTIME_REPLAY Replay saved corrected poses at their recorded scan rate.
% This demonstrates timestamp-paced visualization and models a serial
% per-scan processing budget. It does NOT run the GaRLIO estimator in MATLAB.
%
% From the repository root in MATLAB:
%   addpath('matlab');
%   report = run_realtime_replay( ...
%       'outputs/full_sequence/trajectory_evaluation.mat', 0.168, 1.0);
%
% processingTimeS is an assumed service time per scan. The default 0.168 s
% came from one offline Python full-sequence run on this computer. It is an
% illustrative input, not MATLAB processing time or a target-device result.
% playbackSpeed=1 plays at recorded timestamps; 2 plays twice as fast.

if nargin < 1 || isempty(matPath)
    repoRoot = fileparts(fileparts(mfilename('fullpath')));
    matPath = fullfile(repoRoot, 'outputs', 'full_sequence', ...
        'trajectory_evaluation.mat');
end
if nargin < 2 || isempty(processingTimeS)
    processingTimeS = 0.168;
end
if nargin < 3 || isempty(playbackSpeed)
    playbackSpeed = 1.0;
end
if exist(matPath, 'file') ~= 2
    error('MAT input file not found: %s', matPath);
end
if processingTimeS < 0 || playbackSpeed <= 0
    error('processingTimeS must be non-negative and playbackSpeed positive');
end

data = load(matPath, 'times_corrected', 'positions_corrected');
times = double(data.times_corrected(:));
positions = double(data.positions_corrected);
if size(positions, 1) ~= 3 || size(positions, 2) ~= numel(times)
    error('Expected positions_corrected as 3xN matching times_corrected');
end
if numel(times) < 2 || any(diff(times) <= 0)
    error('Expected at least two strictly increasing scan timestamps');
end

recordedTimes = times - times(1);
scanIntervals = diff(recordedTimes);
n = numel(times);
arrivalTimes = recordedTimes / playbackSpeed;
completionTimes = zeros(n, 1);
queueWait = zeros(n, 1);
deadlineMiss = false(n, 1);
wallLateness = zeros(n, 1);
completionTimes(1) = arrivalTimes(1) + processingTimeS;

for i = 2:n
    serviceStart = max(arrivalTimes(i), completionTimes(i - 1));
    queueWait(i) = serviceStart - arrivalTimes(i);
    completionTimes(i) = serviceStart + processingTimeS;
    deadlineMiss(i - 1) = completionTimes(i - 1) > arrivalTimes(i);
end
deadlineMiss(n) = completionTimes(n) > arrivalTimes(n) + median(diff(arrivalTimes));

fig = figure('Name', 'GaRLIO recorded real-time replay', 'Color', 'w');
ax = axes(fig);
hold(ax, 'on');
grid(ax, 'on');
axis(ax, 'equal');
xlabel(ax, 'XT32 x [m]');
ylabel(ax, 'XT32 y [m]');
title(ax, 'Corrected trajectory replayed at recorded scan timestamps');
pathLine = animatedline(ax, 'Color', [0.1 0.35 0.8], 'LineWidth', 1.5);
currentPoint = plot(ax, positions(1, 1), positions(2, 1), ...
    'o', 'MarkerFaceColor', [0.9 0.25 0.15], 'MarkerEdgeColor', 'none');
status = text(ax, 0.02, 0.98, '', 'Units', 'normalized', ...
    'VerticalAlignment', 'top', 'BackgroundColor', 'w');
xlim(ax, [min(positions(1, :)), max(positions(1, :))] + [-1 1]);
ylim(ax, [min(positions(2, :)), max(positions(2, :))] + [-1 1]);

wallClock = tic;
for i = 1:n
    dueWallTime = arrivalTimes(i);
    remaining = dueWallTime - toc(wallClock);
    if remaining > 0
        pause(remaining);
    end
    actualWallTime = toc(wallClock);
    wallLateness(i) = max(0, actualWallTime - dueWallTime);

    addpoints(pathLine, positions(1, i), positions(2, i));
    set(currentPoint, 'XData', positions(1, i), 'YData', positions(2, i));
    set(status, 'String', sprintf(['scan %d/%d | recorded time %.2f s | ' ...
        'playback lateness %.3f s'], i, n, recordedTimes(i), wallLateness(i)));
    drawnow;
end

report.scan_count = n;
report.recorded_duration_s = recordedTimes(end);
report.mean_scan_interval_s = mean(scanIntervals);
report.median_scan_interval_s = median(scanIntervals);
report.assumed_processing_time_s = processingTimeS;
report.playback_speed = playbackSpeed;
report.simulated_deadline_misses = sum(deadlineMiss);
report.simulated_deadline_miss_fraction = mean(deadlineMiss);
report.max_simulated_queue_wait_s = max(queueWait);
report.mean_simulated_queue_wait_s = mean(queueWait);
report.max_visualization_lateness_s = max(wallLateness);
report.mean_visualization_lateness_s = mean(wallLateness);

fprintf('\nRecorded replay summary\n');
fprintf('Scans: %d; recorded span: %.2f s\n', n, report.recorded_duration_s);
fprintf('Scan interval: mean %.4f s, median %.4f s\n', ...
    report.mean_scan_interval_s, report.median_scan_interval_s);
fprintf('Assumed service time: %.4f s/scan (input, not MATLAB timing)\n', ...
    processingTimeS);
fprintf('Simulated serial deadline misses: %d/%d (%.1f%%)\n', ...
    report.simulated_deadline_misses, n, ...
    100 * report.simulated_deadline_miss_fraction);
fprintf('Maximum simulated queue wait: %.3f s\n', ...
    report.max_simulated_queue_wait_s);
fprintf('Visualization lateness: mean %.3f s, max %.3f s\n', ...
    report.mean_visualization_lateness_s, report.max_visualization_lateness_s);
fprintf(['Interpretation: only recorded-pose playback ran in MATLAB. ' ...
    'Deadline results use the assumed service time and do not measure ' ...
    'a MATLAB estimator or live sensors.\n']);
end
