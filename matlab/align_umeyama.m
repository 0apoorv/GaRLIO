function [alignedSource, R, t] = align_umeyama(source, target)
%ALIGN_UMEYAMA Rigidly align 3xN source points to 3xN target points.
% This is the same position-only, no-scale Umeyama alignment used in Python.

if size(source, 1) ~= 3 || ~isequal(size(source), size(target))
    error('source and target must have the same 3xN shape');
end
n = size(source, 2);
if n == 0
    error('at least one point is required for alignment');
end

muSource = mean(source, 2);
muTarget = mean(target, 2);
sourceCentered = source - muSource;
targetCentered = target - muTarget;
covariance = (targetCentered * sourceCentered') / n;
[U, ~, V] = svd(covariance);
S = eye(3);
if det(U * V') < 0
    S(3, 3) = -1;
end
R = U * S * V';
t = muTarget - R * muSource;
alignedSource = R * source + t;
end
