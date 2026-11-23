function report = generate_sensor_event_stream(sourcePath, outputPath)
%GENERATE_SENSOR_EVENT_STREAM Merge raw sensor timestamps into event order.
% The source MAT contains IMU, radar, and LiDAR measurements exported from the
% Snail-Radar data. Event type codes: 1=IMU, 2=radar, 3=LiDAR.
%
% In MATLAB Online, pass paths in MATLAB Drive, for example:
%   report = generate_sensor_event_stream( ...
%       'sensor_stream_source.mat', 'matlab_generated_event_stream.mat');

if nargin < 1 || isempty(sourcePath)
    sourcePath = 'sensor_stream_source.mat';
end
if nargin < 2 || isempty(outputPath)
    outputPath = 'matlab_generated_event_stream.mat';
end
if exist(sourcePath, 'file') ~= 2
    error('Sensor input MAT file not found: %s', sourcePath);
end

source = load(sourcePath, 'imu_times', 'radar_times', 'lidar_times');
imuTimes = double(source.imu_times(:));
radarTimes = double(source.radar_times(:));
lidarTimes = double(source.lidar_times(:));

eventTimes = [imuTimes; radarTimes; lidarTimes];
eventTypes = [ones(numel(imuTimes), 1, 'uint8'); ...
              2 * ones(numel(radarTimes), 1, 'uint8'); ...
              3 * ones(numel(lidarTimes), 1, 'uint8')];
eventIndices = uint32([(1:numel(imuTimes))'; ...
                       (1:numel(radarTimes))'; ...
                       (1:numel(lidarTimes))']);

% At equal timestamps, process IMU, then radar, then LiDAR. This ensures the
% estimator has measurements at the scan timestamp before a LiDAR update.
ordered = sortrows([eventTimes, double(eventTypes), double(eventIndices)], [1 2 3]);
eventTimes = ordered(:, 1)';
eventTypes = uint8(ordered(:, 2)');
eventIndices = uint32(ordered(:, 3)');
clear ordered source

save(outputPath, 'eventTimes', 'eventTypes', 'eventIndices', '-v7');
report = struct( ...
    'event_count', numel(eventTimes), ...
    'imu_count', numel(imuTimes), ...
    'radar_count', numel(radarTimes), ...
    'lidar_count', numel(lidarTimes), ...
    'first_time_s', min(eventTimes), ...
    'last_time_s', max(eventTimes), ...
    'duration_s', max(eventTimes) - min(eventTimes), ...
    'output_file', outputPath);
report.imu_median_period_s = median(diff(imuTimes));
report.radar_median_period_s = median(diff(radarTimes));
report.lidar_median_period_s = median(diff(lidarTimes));
report.imu_median_rate_hz = 1 / report.imu_median_period_s;
report.radar_median_rate_hz = 1 / report.radar_median_period_s;
report.lidar_median_rate_hz = 1 / report.lidar_median_period_s;
fprintf('Generated %d timestamp-ordered events: %d IMU, %d radar, %d LiDAR.\n', ...
    report.event_count, report.imu_count, report.radar_count, report.lidar_count);
fprintf('Event span %.3f s. Saved %s\n', report.duration_s, outputPath);
fprintf(['Median rates from source timestamps: IMU %.1f Hz, radar %.1f Hz, ' ...
         'LiDAR %.1f Hz.\n'], report.imu_median_rate_hz, ...
    report.radar_median_rate_hz, report.lidar_median_rate_hz);
end
