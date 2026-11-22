function result = compute_relative_errors(estTimes, estPositions, estRotations, ...
        gtTimes, gtPositions, gtRotations, intervalS, maxDt, intervalToleranceS)
%COMPUTE_RELATIVE_ERRORS Compute fixed-time RTE and RRE from matched poses.
% Times are in seconds, positions are 3xN in metres, and rotations are 3x3xN.
% Each R maps body-frame coordinates into the world frame (T_WB).
% Relative translation is represented in the start pose's body frame. No global
% alignment is applied. RRE is returned as a geodesic angle in radians/degrees.

if nargin < 8 || isempty(maxDt)
    maxDt = 0.05;
end
if nargin < 9 || isempty(intervalToleranceS)
    intervalToleranceS = 0.05;
end
estTimes = estTimes(:)';
gtTimes = gtTimes(:)';
if intervalS <= 0
    error('intervalS must be positive');
end
if maxDt < 0 || intervalToleranceS < 0
    error('timestamp tolerances must be non-negative');
end
if size(estPositions, 1) ~= 3 || size(estPositions, 2) ~= numel(estTimes)
    error('estPositions must be 3xN and match estTimes');
end
if size(gtPositions, 1) ~= 3 || size(gtPositions, 2) ~= numel(gtTimes)
    error('gtPositions must be 3xN and match gtTimes');
end
if size(estRotations, 1) ~= 3 || size(estRotations, 2) ~= 3 || ...
        size(estRotations, 3) ~= numel(estTimes)
    error('estRotations must be 3x3xN and match estTimes');
end
if size(gtRotations, 1) ~= 3 || size(gtRotations, 2) ~= 3 || ...
        size(gtRotations, 3) ~= numel(gtTimes)
    error('gtRotations must be 3x3xN and match gtTimes');
end
if isempty(gtTimes)
    error('ground-truth timestamps must not be empty');
end

matchedEstIdx = [];
matchedGtIdx = [];
matchedTimes = [];
for i = 1:numel(estTimes)
    [dt, j] = min(abs(gtTimes - estTimes(i)));
    if dt <= maxDt
        matchedEstIdx(end + 1) = i; %#ok<AGROW>
        matchedGtIdx(end + 1) = j; %#ok<AGROW>
        matchedTimes(end + 1) = estTimes(i); %#ok<AGROW>
    end
end

translationErrors = [];
rotationErrorsRad = [];
pairTimes = zeros(2, 0);
pairDurations = [];
for startIdx = 1:numel(matchedTimes)
    tStart = matchedTimes(startIdx);
    later = find(matchedTimes(startIdx + 1:end) > tStart) + startIdx;
    if isempty(later)
        continue;
    end
    targetTime = tStart + intervalS;
    [~, nearestOffset] = min(abs(matchedTimes(later) - targetTime));
    endIdx = later(nearestOffset);
    duration = matchedTimes(endIdx) - tStart;
    if abs(duration - intervalS) > intervalToleranceS
        continue;
    end

    i = matchedEstIdx(startIdx);
    j = matchedEstIdx(endIdx);
    gi = matchedGtIdx(startIdx);
    gj = matchedGtIdx(endIdx);

    Rgt = gtRotations(:, :, gi)' * gtRotations(:, :, gj);
    Rest = estRotations(:, :, i)' * estRotations(:, :, j);
    tgt = gtRotations(:, :, gi)' * (gtPositions(:, gj) - gtPositions(:, gi));
    test = estRotations(:, :, i)' * (estPositions(:, j) - estPositions(:, i));

    Rerror = Rgt' * Rest;
    terror = Rgt' * (test - tgt);
    cosAngle = min(1, max(-1, (trace(Rerror) - 1) / 2));
    veeAntisymmetric = [Rerror(3, 2) - Rerror(2, 3); ...
        Rerror(1, 3) - Rerror(3, 1); ...
        Rerror(2, 1) - Rerror(1, 2)];
    sinAngle = 0.5 * norm(veeAntisymmetric);

    translationErrors(end + 1) = norm(terror); %#ok<AGROW>
    rotationErrorsRad(end + 1) = atan2(sinAngle, cosAngle); %#ok<AGROW>
    pairTimes(:, end + 1) = [tStart; matchedTimes(endIdx)]; %#ok<AGROW>
    pairDurations(end + 1) = duration; %#ok<AGROW>
end

if isempty(translationErrors)
    error(['No valid pose pairs for interval %.6g s (tolerance %.6g s); ' ...
        '%d poses matched within %.6g s'], ...
        intervalS, intervalToleranceS, numel(matchedTimes), maxDt);
end

result.n_matched = numel(matchedTimes);
result.n_pairs = numel(translationErrors);
result.interval_s = intervalS;
result.translation_errors = translationErrors;
result.translation_rmse = sqrt(mean(translationErrors.^2));
result.translation_mean = mean(translationErrors);
result.translation_median = median(translationErrors);
result.translation_max = max(translationErrors);
result.rotation_errors_rad = rotationErrorsRad;
result.rotation_errors_deg = rotationErrorsRad * (180 / pi);
result.rotation_rmse_rad = sqrt(mean(rotationErrorsRad.^2));
result.rotation_rmse_deg = result.rotation_rmse_rad * (180 / pi);
result.rotation_mean_rad = mean(rotationErrorsRad);
result.rotation_mean_deg = result.rotation_mean_rad * (180 / pi);
result.pair_times = pairTimes;
result.pair_durations = pairDurations;
end
