function result = compute_ate(estTimes, estPositions, gtTimes, gtPositions, maxDt)
%COMPUTE_ATE Match positions by time, rigidly align, and compute ATE.
% Times are row/column vectors in seconds; positions are 3xN in metres.
% Alignment is rigid (rotation + translation), without scale, matching Python.

if nargin < 5
    maxDt = 0.05;
end
estTimes = estTimes(:)';
gtTimes = gtTimes(:)';
if size(estPositions, 1) ~= 3 || size(estPositions, 2) ~= numel(estTimes)
    error('estPositions must be 3xN and match estTimes');
end
if size(gtPositions, 1) ~= 3 || size(gtPositions, 2) ~= numel(gtTimes)
    error('gtPositions must be 3xN and match gtTimes');
end
if isempty(gtTimes)
    error('ground-truth timestamps must not be empty');
end

estIdx = [];
gtIdx = [];
matchedTimes = [];
for i = 1:numel(estTimes)
    [dt, j] = min(abs(gtTimes - estTimes(i)));
    if dt <= maxDt
        estIdx(end + 1) = i; %#ok<AGROW>
        gtIdx(end + 1) = j; %#ok<AGROW>
        matchedTimes(end + 1) = estTimes(i); %#ok<AGROW>
    end
end

if numel(estIdx) < 3
    error('Only %d matched points (need >=3 for alignment)', numel(estIdx));
end
matchedEst = estPositions(:, estIdx);
matchedGt = gtPositions(:, gtIdx);
[alignedEst, R, t] = align_umeyama(matchedEst, matchedGt);
errors = sqrt(sum((alignedEst - matchedGt).^2, 1));

result.n_matched = numel(estIdx);
result.rmse = sqrt(mean(errors.^2));
result.mean = mean(errors);
result.median = median(errors);
result.max = max(errors);
result.aligned_est = alignedEst;
result.matched_gt = matchedGt;
result.matched_times = matchedTimes;
result.errors = errors;
result.R = R;
result.t = t;
result.est_indices = estIdx;
result.gt_indices = gtIdx;
end
