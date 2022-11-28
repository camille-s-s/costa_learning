function [exp_data, outliers, arrayUnit, arrayRgns, fixOnInds, stimTimeInds, nTrls, nTrlsPerSet, nSets, setID] ...
    = preprocess_data_for_RNN_training(allSpikes, allEvents, T, ...
    doSmooth, rmvOutliers, meanSubtract, doSoftNorm, ...
    smoothWidth, dtData, arrayUnit, arrayRgns)

exp_data = allSpikes;

% cleaning: smooth with gaussian
if doSmooth
    % exp_data = smoothdata(exp_data, 2, 'gaussian', smoothWidth / dtData); % convert smoothing kernel from sec to #bins);
    exp_data = smooth_data(exp_data', dtData, smoothWidth); exp_data = exp_data'; % try direct convolution custom not matlab toolbox
end

% cleaning: outlier removal
if rmvOutliers
    
    outliers = isoutlier(mean(exp_data, 2), 'percentiles', [10 100]);
    exp_data = exp_data(~outliers, :);
    arrayUnit = arrayUnit(~outliers, :);
    
    % update indexing vectors
    for iRgn = 1 : size(arrayRgns, 1)
        arrayRgns{iRgn, 3}(outliers) = [];
    end
    
else
    outliers = [];
end

% transformation: center each neuron by subtracting its mean activity
if meanSubtract
    meanTarg = exp_data - mean(exp_data, 2);
    exp_data = meanTarg;
end

% transformation: this will soft normalize a la Churchland papers (so all
% activity is on roughly similar scale)
if doSoftNorm
    normfac = range(exp_data, 2); % + (dtData * 10); % normalization factor = firing rate range + alpha
    exp_data = exp_data ./ normfac;
end

% housekeeping
if any(isnan(exp_data(:)))
    keyboard
end

% pull trial starts
fixOnInds = [find(allEvents == 1), size(exp_data, 2)];

% pull event labels by trial (in time relative to start of each trial)
stimTimeInds = find(allEvents == 2) - find(allEvents == 1); % (1 = fixation, 2 = stim, 3 = choice, 4 =  outcome, 5 = time of next trl fixation)

% sanity check match in #trials
assert(isequal(sum(allEvents == 1), height(T)))

% get block/set structure (s sets of j trials each)
nTrls =  height(T); % height(T); % TO DO: CHANGE THIS BACK AFTER DEV!
nTrlsPerSet = diff([find(T.trls_since_nov_stim == 0); height(T) + 1]); % 2022/03/16 edit
nSets = sum(T.trls_since_nov_stim == 0); % 2022/03/16 edit
setID = repelem(1 : nSets, nTrlsPerSet)'; % 2022/03/16 edit

end