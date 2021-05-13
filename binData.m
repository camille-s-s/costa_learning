function binnedX = binData(X, params)
% Sliding bin for 3D input data. Assumes time is last dimension.

binDim = ndims(X);

% params
sliding         = true; % discrete if false
step            = 25; % (steps to take for each bin)
windowWidth     = size(X, binDim); % size of whole time dimension
binWidth        = 150;

% overwrite defaults based on inputs
if exist('params','var')
    assignParams(who,params);
end

% Determine number of bins.
if sliding
    assert(~isempty(step), ...
           "'step' must be specified if 'sliding' is set to true.")
    binStarts = 1 : step : windowWidth - binWidth + step;  % (overlapping indices for start of bin)
    nBins = length(binStarts);
else
    assert(mod(windowWidth, binWidth) == 0, ...
           ['Discrete bins requested, but specified window width ' ...
            'cannot be evenly divided by specified bin width.'])
    binStarts = 1 : binWidth : windowWidth; % non-overlapping indices for start of bin
    nBins = windowWidth ./ binWidth;
    assert(length(binStarts) == nBins)
end

% Define anonymous function for binning
slideBin = @(X, binStarts, binWidth, iBin, binDim) mean(X(:, :, binStarts(iBin) : binStarts(iBin) - 1 + binWidth), binDim);

% Preallocate for subsequent loop.
binnedX = NaN([size(X, [1 : binDim - 1]), nBins]);

for iBin = 1:nBins
    binnedX(:, :, iBin) = slideBin(X, binStarts, binWidth, iBin, binDim);
end

end