% make plots of RNN outputs, currently for consecutive trials starting from
% the top
%
% 1. Top 3 PCs from PCA of all activity (all arrays) over nD + 1
% consecutive trials from setToPlot.

clearvars;
close all

% params
aFac            = 0.01; % idk comes from softNormalize > o3_AddCurrentsToTD_runs
center_data     = true;
n_PCs           = 10;
doSmooth        = true;
nDeltasToPlot   = 5; % for subsampling over whole session
nD              = 10; % for within one session (assumed to be <= nTrlsPerSet)
nSamples        = 21; % for within one trial
setToPlot       = 2; % only one for now - must be 2 or greater to get last trial prev set
figVer          = 'v3test'; %'v3test'; % ''; % 'v3test/'; % or ''

% in and outdirs
bd              = '~/Dropbox (BrAINY Crew)/costa_learning/';
mouseVer        = ['PINKY_VERSION/'];
mdlDir          = [bd 'models/', mouseVer, figVer, filesep];
RNNfigdir       = [bd 'figures/', mouseVer];
RNNSampleDir    = [mdlDir 'model_samples/'];
spikeInfoPath   = [bd 'reformatted_data/'];

%%
addpath(genpath(bd))

if ~isfolder(RNNSampleDir)
    mkdir(RNNSampleDir)
end

cd(spikeInfoPath)

allFiles = dir('*_meta.mat');
bad_files = arrayfun(@(iFile) any(strfind(allFiles(iFile).name, '._')), 1:length(allFiles));
allFiles(bad_files) = [];

desiredOrder = {'left_cdLPFC', 'right_cdLPFC', ...
    'left_mdLPFC', 'right_mdLPFC', ...
    'left_vLPFC', 'right_vLPFC', ...
    'left_rdLPFC','right_rdLPFC'};

if setToPlot < 2
    disp('set # must be 2 or greater!')
    keyboard
end

if strcmp (computer, 'MACI64')
    fontSzS = 10;
    fontSzM = 14;
    fontSzL = 16;
elseif strcmp(computer, 'GLNXA64')
    fontSzS = 6;
    fontSzM = 8;
    fontSzL = 9;
end

for iFile = 1 : 3 %  : 3 %4 : 6 % 6 : length(allFiles) % 1 : length(allFiles) % for each session....
    
    fName = allFiles(iFile).name;
    fID = fName(1:strfind(fName, '_') - 1);
    monkey = fID(1);
    ssnDate = fID(2:end);
    cd([mdlDir, monkey, ssnDate, filesep])
    currSsn = dir('rnn_*_set*_trial*.mat');
    
    if strcmp(figVer, 'v3test')
        figNameStem = [monkey, ssnDate, '_v3test']; % line that makes this backwards compatible with makeRNNPlots
    elseif strcmp(figVer, '')
        figNameStem = [monkey, ssnDate];
    else
        disp('check if v3test directory is consistently named!')
        keyboard
    end
    
      allTrialIDs = unique(arrayfun(@(i) ...
        str2double(currSsn(i).name(strfind(currSsn(i).name,'trial') + 5 : end - 4)), 1:length(currSsn)));
    
    nTrls = length(allTrialIDs);

    % for first trial of a session, pull params (this field is empty for
    % all other trials in a session)
    firstTrl = find(arrayfun(@(i) isequal(currSsn(i).name, ['rnn_', monkey, ssnDate, '_set1_trial1.mat']), 1 : nTrls));
    if isempty(firstTrl) % due to old naming convention
        firstTrl = find(arrayfun(@(i) isequal(currSsn(i).name, ['rnn_', monkey, ssnDate, '_set0_trial1.mat']), 1 : nTrls));
    end
    
    spikeInfoName = [currSsn(firstTrl).name(5 : median(strfind(currSsn(firstTrl).name, '_'))-1), '_meta.mat'];
    load([spikeInfoPath, spikeInfoName], 'spikeInfo', 'trlInfo', 'dsAllEvents')
    load([mdlDir, monkey, ssnDate, filesep, currSsn(firstTrl).name]);
    binSize = RNN.mdl.dtData; % in sec
    tDataInit = RNN.mdl.tData;
    tRNNInit = RNN.mdl.tRNN;
 
    if strcmp(figVer, 'v3test')
        assert(RNN.mdl.params.doSmooth == 1 & ...
            RNN.mdl.params.doSoftNorm == 1 & ....
            RNN.mdl.params.normByRegion == 0 & ...
            RNN.mdl.params.rmvOutliers == 1 & ...
            RNN.mdl.params.dtFactor == 20 & ...
            RNN.mdl.params.g == 1.5 & ...
            RNN.mdl.params.tauRNN == 0.001 & ...
            RNN.mdl.params.tauWN == 0.1 & ...
            RNN.mdl.params.ampInWN == 0.001 & ...
            RNN.mdl.params.nRunTot == 1010) % for versions from 2022 onward (tentatively)
    elseif strcmp(figVer, '')
        assert(RNN.mdl.params.doSmooth == 1 & ...
            RNN.mdl.params.doSoftNorm == 1 & ....
            RNN.mdl.params.normByRegion == 0 & ...
            RNN.mdl.params.rmvOutliers == 1 & ...
            RNN.mdl.params.dtFactor == 20 & ...
            RNN.mdl.params.g == 1.5 & ...
            RNN.mdl.params.tauRNN == 0.001 & ...
            RNN.mdl.params.tauWN == 0.1 & ...
            RNN.mdl.params.ampInWN == 0.001 & ...
            RNN.mdl.params.nRunTot == 505) % for versions from 2022 onward (tentatively)
    end
    
    if isfield(RNN.mdl.params, 'smoothWidth')
        assert(RNN.mdl.params.smoothWidth == 0.15)
    end
    
    % load full targets that went into fitCostaRNN for sanity checking with
    % targets from each trial's model
    metafnm = [spikeInfoPath, monkey, ssnDate, '_spikesCont'];
    S = load(metafnm);
    targets = S.dsAllSpikes;
    
    % pull event labels by trial (in time relative to start of each trial)
    stimTimeInds = find(dsAllEvents == 2) - find(dsAllEvents == 1);
    
    %% recreate target prep from fitCostaRNN for comparison of targets from each trial
    
    % cleaning: smooth with gaussian
    if RNN.mdl.params.doSmooth
        targets = smoothdata(targets, 2, 'gaussian', 0.15 / RNN.mdl.dtData); % convert smoothing kernel from msec to #bins);
    end
    
    % cleaning: outlier removal
    if RNN.mdl.params.rmvOutliers
        
        % outliers = isoutlier(mean(targets, 2), 'percentiles', [0.5 99.5]);
        outliers = isoutlier(mean(targets, 2), 'percentiles', [1 99]); % version as of 2022
        targets = targets(~outliers, :);
        
        if size(spikeInfo, 1) ~= size(targets, 1)
            spikeInfo = spikeInfo(~outliers, :);
        end
        
        % update indexing vectors
        arrayRgns = RNN.mdl.params.arrayRegions;
        for iRgn = 1 : size(arrayRgns, 1)
            if size(arrayRgns{iRgn, 3}, 1) ~= size(targets, 1)
                disp('array rgns in param struct should match outlier removal numbers...')
                keyboard
                arrayRgns{iRgn, 3}(outliers) = [];
            end
        end
    end
    
    % transformation: this will soft normalize a la Churchland papers
    if RNN.mdl.params.doSoftNorm
        normfac = range(targets, 2); % + (dtData * 10); % normalization factor = firing rate range + alpha
        targets = targets ./ normfac;
    end
    
    % housekeeping
    if any(isnan(targets(:)))
        keyboard
    end
    
    % generate setID label from trlInfo table
    nTrlsPerSet = diff([find(trlInfo.trls_since_nov_stim == 0); height(trlInfo) + 1]); % 2022/03/16 edit
    nSets = sum(trlInfo.trls_since_nov_stim == 0); % 2022/03/16 edit
    setID = repelem(1:nSets, nTrlsPerSet)'; % 2022/03/16 edit
    allSetIDs = unique(setID);
    
    maxCompleteSet = find(nTrls < cumsum(nTrlsPerSet), 1, 'first') - 1;
    
    if ~isempty(maxCompleteSet) % will be empty if complete!
        nTrls = find(setID == maxCompleteSet, 1, 'last'); % so you can plot incomplete sets
    end
    %% collect data
    
    % set up indexing vectors for submatrices
    rgns            = RNN.mdl.params.arrayRegions;
    dtData          = RNN.mdl.dtData;
    arrayList       = rgns(:, 2);
    nRegions        = length(arrayList);
    rgnColors       = [1 0 0; 1 0 0; 1 0 1; 1 0 1; 0 0 1; 0 0 1; 0 1 0; 0 1 0]; % r, m, b, g brewermap(nRegions, 'Spectral');% cmap(round(linspace(1, 255, nRegions)),:);
    rgnLinestyle    = {'-', '-.', '-', '-.', '-', '-.', '-', '-.'};
    JDim            = size(RNN.mdl.J);
    fittedConsJDim  = size(RNN.mdl.fittedConsJ);
    assert(isequal(fittedConsJDim(end), nSamples))
    
    clear RNN
    
    % collect outputs for all trials in a session
    trlNum              = NaN(1, nTrls);
    R_U                 = cell(nTrls, 1);
    R_S                 = cell(nTrls, 1);
    J_U                 = NaN([JDim, nTrls]);
    J_S                 = NaN([JDim, nTrls]);
    J0_U                = NaN([JDim, nTrls]);
    J0_S                = NaN([JDim, nTrls]);
    D_U                 = cell(nTrls, 1);
    D_S                 = cell(nTrls, 1);
    pVarsTrls           = zeros(nTrls, 1);
    chi2Trls            = zeros(nTrls, 1);
    fittedJ             = cell(nTrls, 1);
    
    tic
    
    activitySampleAll      = cell(nTrls, 1);
    intraRgnJSampleAll  = cell(nTrls, 1);
    allSPTimePoints     = NaN(nTrls, nSamples);
    
    % load 'targets', 'fixOnInds', 'allPossTS', 'nTrls'
    if isfolder([mdlDir 'targets' filesep]) % this directory exists for v3, not previous versions
        load([mdlDir 'targets' filesep, 'rnn_', monkey, ssnDate, '_targets'], 'targets', 'fixOnInds', 'allPossTS');
    end
    
    for i = 1 : nTrls
        
        mdlfnm              = dir([mdlDir, monkey, ssnDate, filesep, 'rnn_', monkey, ssnDate, '_set*_trial', num2str(i), '.mat']);
        tmp                 = load(mdlfnm.name);
        mdl                 = tmp.RNN.mdl;
        
        assert(mdl.dtRNN == 0.0005 & ...
            mdl. dtData == 0.01)
        
        iTarget             = mdl.iTarget;
        
        % turn a scrambly boy back into its OG order - target_unit_order(iTarget) = R_and_J_unit_order
        % aka targets and regions are unscrambled, R and J are scrambled
        
        [~, unscramble] = sort(iTarget); % Gives you indices from scrambled to unscrambled
        
        trlNum(i)           = mdl.iTrl;
        R_S{i}              = mdl.RMdlSample;
        J_S(:, :, i)        = mdl.J;
        J0_S(:, :, i)       = mdl.J0;
        
        if isfolder([mdlDir 'targets' filesep])
            % generate targets separately (they don't get saved with the model
            % in fitCostaRNNv3)
            iStart = fixOnInds(i); % start of trial
            iStop = fixOnInds(i + 1) - 1; % right before start of next trial
            currTargets = targets(:, iStart:iStop);
            % tData = allPossTS(iStart:iStop); % timeVec for current data
            % tRNN = tData(1) : dtRNN : tData(end); % timevec for RNN
            
            D_S{i}              = currTargets(iTarget, :);
            D_U{i}              = currTargets;
            assert(isequal(D_U{i}, D_S{i}(unscramble, :))); % indices from scrambled back to original
        else
            D_S{i}              = mdl.targets(iTarget, :);
            D_U{i}              = mdl.targets;
            assert(isequal(mdl.targets, D_S{i}(unscramble, :))); % indices from scrambled back to original
        end
            
        % collect first and last to assess convergence
        if i == 1
            firstPVars      = mdl.pVars;
            firstChi2       = mdl.chi2;
        end
        
        if i == nTrls
            lastPVars       = mdl.pVars;
            lastChi2        = mdl.chi2;
        end
     
        pVarsTrls(i)        = mdl.pVars(end);
        chi2Trls(i)         = mdl.chi2(end);
        
        % in the event of permuted iTarget
        J_U(:, :, i)        = mdl.J(unscramble, unscramble);
        J0_U(:, :, i)       = mdl.J0(unscramble, unscramble);
        R_U{i}              = mdl.RMdlSample(unscramble, :); % since targets(iTarget) == R
       
        allSPTimePoints(i, :) = mdl.sampleTimePoints;
        
    end
    
    toc
    
    if sum(pVarsTrls < 0) >= 1
        disp(['running error for this session starting at trl ', num2str(find(pVarsTrls<0, 1)), '! ...'])
    end
    
    assert(all(arrayfun(@(t) isequal(J0_U(:,:,t+1),J_U(:,:,t)), 1 : nTrls-1)))
    
    %% HOUSEKEEPING: SPATIAL
    
    % make regions more legible
    rgnOrder = arrayfun(@(iRgn) find(strcmp(rgns(:,1), desiredOrder{iRgn})), 1 : nRegions); % ensure same order across sessions
    rgns(:, 1) = arrayfun(@(iRgn) strrep(rgns{iRgn, 1}, 'left_', 'L'), 1 : nRegions, 'un', false)';
    rgns(:, 1) = arrayfun(@(iRgn) strrep(rgns{iRgn, 1}, 'right_', 'R'), 1 : nRegions, 'un', false)';
    rgns = rgns(rgnOrder, :);
    rgnLabels = rgns(:, 1);
    rgnLabels = arrayfun(@(iRgn) rgnLabels{iRgn}(1:end-3), 1:nRegions, 'un', false);
    rgnIxToPlot = cell2mat(arrayfun(@(iRgn) contains(rgnLabels{iRgn}(1), 'L'), 1 : nRegions, 'un', false));
    
    % id bad rgns
    inArrays = rgns(:, 3);
    
    % rgn w really low # units gets excluded
    badRgn = arrayfun(@(iRgn) sum(inArrays{iRgn}) < n_PCs, 1:nRegions);
    
    %% HOUSEKEEPING: TEMPORAL
    
    % reorder by ascending trlNum
    [~, trlSort] = sort(trlNum, 'ascend');
    
    % quick check to make sure no consecutive trials are skipped!
    assert(numel(unique(diff(trlNum(trlSort)))) == 1)
    
    % if pVar glitched into negatives at some trial, only plot trls up to the last good
    % trial
    pVarsTrls = pVarsTrls(trlSort);
    
    if sum(pVarsTrls < 0) > 20 % if more than one in a row....
        lastGoodTrl = find(pVarsTrls<0, 1) - 1;
        trlSort = trlSort(1 : lastGoodTrl);
        setID = setID(1 : lastGoodTrl);
        
        % truncate a bit more to last complete set
        trlNumLastGoodSet = find(setID == max(setID) - 1, 1, 'last');
        trlSort = trlSort(1 : trlNumLastGoodSet);
        nTrls = trlNumLastGoodSet;
    end
    
    D_S = D_S(trlSort); %  d = cell2mat(D_U'); isequal(d(iTarget, :), cell2mat(D_S'))
    D_U = D_U(trlSort);
    J_S = J_S(:, :, trlSort); % j = squeeze(mean(J_U, 3)), isequal(j(iTarget, iTarget), squeeze(mean(J_S, 3)))
    J0_S = J0_S(:, :, trlSort);
    R_S = R_S(trlSort); % r = cell2mat(R_U'); isequal(r(iTarget, :), cell2mat(R_S'))
    
    % NOTE: USE unscrambled J/J0/R IF YOU WANNA USE SPIKEINFO TO INDEX!
    J_U = J_U(:, :, trlSort);
    J0_U = J0_U(:, :, trlSort);
    R_U = R_U(trlSort);
    
    allSPTimePoints = allSPTimePoints(trlSort, :);
    % sanity check that all J0s are different
    J0_resh = cell2mat(arrayfun(@(iTrl) ...
        reshape(squeeze(J0_U(:, :, iTrl)), 1, size(J0_U, 1)^2), 1 : size(J0_U, 3), 'un', false)');
    assert(isequal(size(unique(J0_resh, 'rows'), 1), nTrls))
    
    % compute some useful quantities. note that _U is in same unit order as
    % spikeInfo
    trlDursRNN = arrayfun(@(iTrl) size(D_U{iTrl}, 2), 1 : nTrls)';
    [~, ia, ic] = unique(setID(setID ~= 1)');
    trlsPerSetRNN = accumarray(ic, 1);
    minTrlsPerSet = min(trlsPerSetRNN(:));
    DFull = cell2mat(D_U');
    
    % sanity check #1: make sure we are where we think we are in trlInfo
    % (for plotting consecutive trials (last trial prev set is first - to
    % compare delta between sets) as well as delta within set) using number
    % of trials using trial/set structure
    trl = trlInfo.trls_since_nov_stim;
    tmpLast = [find(trl == 0) - 1; height(trlInfo)];
    tmpLast(tmpLast == 0) = []; % delete first  trial
    trlsPerSetTbl = trl(tmpLast) + 1;
    n = numel(trlsPerSetRNN);
    matchIx = find(arrayfun(@(i) isequal(trlsPerSetRNN(1 : end), trlsPerSetTbl(i : i + n - 1)), ...
        1 : numel(trlsPerSetTbl) - n)); % TO DO NEXT: USE THIS IN LATER PLOTTING!
    
    if isempty(matchIx)
        disp('might have incomplete set or not enough trials, trying xcorr comparison...')
        [r, lags] = xcorr(trlsPerSetTbl, trlsPerSetRNN);
        matchIx = lags(r == max(r));
        if isempty(matchIx)
            disp('might have incomplete set or not enough trials or other error. ')
            keyboard
        end
    end
    
    % sanity check #2: compare targets (FULL TARGETS, TRIALS CONCATENATED
    % END TO END) with the snippets from each model .mat file
    fixOnInds = [find(dsAllEvents == 1), size(targets, 2)];
    truncTarg = targets(:, fixOnInds(1) : fixOnInds(1) + size(DFull, 2) - 1);
    assert(isequal(truncTarg, DFull))
    
    % sanity check #3: make sure that RNN trial indexing matches up to
    % table (using trial durations)
    trlDursTbl = round(diff(round(trlInfo.event_times(:, 1), 3, 'decimals')) / dtData);
    [all_C, all_lags] = xcorr(trlDursRNN - mean(trlDursRNN), trlDursTbl - mean(trlDursTbl), 100);
    best_lag = all_lags(all_C==abs(max(all_C)));
    
    try
        assert(best_lag == 0) % trial durations between trlInfo table and trlDurs from RNN snippets should match
    catch
        keyboard
    end
    
    % for demarcating new trials in later plotting
    nSPFull = size(DFull,2);
    nSPTmp = cell2mat(cellfun(@size, D_U, 'un', 0));
    nSP = nSPTmp(:, 2);
    newTrlInds = [1; cumsum(nSP(1:end-1))+1];
    trlStartsTmp = [newTrlInds - 1; size(DFull, 2)];
    trlStarts = cell2mat(arrayfun(@(iTrl) [trlStartsTmp(iTrl, 1)+1 trlStartsTmp(iTrl+1, 1)], 1:nTrls, 'un', 0)');
    
    % for trial averaging from all available trials (get the shortest trial and
    % then trim data from each trial to be as long as that one, then take
    % average over all those trials (for projecting onto)
    shortestTrl = min(trlDursRNN(:));
    DTrunc = D_U(:);
    DTrunc = arrayfun(@(iTrl) DTrunc{iTrl}(:, 1 : shortestTrl), 1 : nD + 1, 'un', false)';
    DTrunc = cell2mat(DTrunc');
    DMean = cell2mat(arrayfun(@(n) mean(reshape(DTrunc(n, :), shortestTrl, size(DTrunc, 2)/shortestTrl), 2)', 1 : size(DTrunc, 1), 'un', false)');

    %% get inds to order J by region
    
    % get original unit order (as trained) and reorder so within-rgn is on-diagonal and between-rgn is off-diagonal
    count = 1;
    nUnitsAll = NaN(nRegions, 1);
    newOrder = [];
    
    for iRgn = 1 : nRegions
        in_rgn = rgns{iRgn,3};
        newOrder = [newOrder; find(in_rgn)]; % reorder J so that rgns occur in order
        nUnitsRgn = sum(in_rgn);
        nUnitsAll(iRgn) = nUnitsRgn;
    end
    
    % J is reordered by region, so R_U must be as well
    R_U_reordered = arrayfun(@(iTrl) R_U{iTrl}(newOrder, :), 1 : nTrls, 'un', 0)';
    J_U_reordered = J_U(newOrder, newOrder, :); % of note: J_U == J_S
    
    %% reorder full J and use it to get intra/inter only Js + plot J, R distributions + CURBD
      
    histFigDir = [RNNfigdir, 'histograms', filesep]; % pVarOverTrls, first trl convergence, last trl convergence
    if ~isfolder(histFigDir)
        mkdir(histFigDir)
    end
    
    CURBDFigDir = [RNNfigdir, 'CURBD', filesep]; % pVarOverTrls, first trl convergence, last trl convergence
    if ~isfolder(CURBDFigDir)
        mkdir(CURBDFigDir)
    end
    
    % get inds for new order by region
    tmp = [0; nUnitsAll];
    JLblPos = arrayfun(@(iRgn) sum(tmp(1 : iRgn-1)) + tmp(iRgn)/2, 2 : nRegions); % for the labels separating submatrices
    JLinePos = cumsum(nUnitsAll)'; % for the lines separating regions
    newRgnInds = [0, JLinePos];
    nUnits = sum(nUnitsAll);
    activitySampleAll = NaN(nUnits, nSamples, nTrls);
    
    % get region labels for CURBD
    curbdRgns = [rgns(:, 1), arrayfun(@(i) newRgnInds(i) + 1 : newRgnInds(i + 1), 1 : nRegions, 'un', 0)'];

    % get sample activvities
    for iTrl = 1 : nTrls
        sampleTimePoints = allSPTimePoints(iTrl, :);
        activitySampleAll(:, :, iTrl) = R_U_reordered{iTrl}(:, sampleTimePoints);
    end
    
    % save J submatrices by set
    interJOverTrls = NaN(nTrls, 1);
    intraJOverTrls = NaN(nTrls, 1);
    fullJOverTrls = NaN(nTrls, 1);
    
    % for pulling trials from within a fixed time duration
    ts = trlInfo.event_times;
    lastTrlWithinHr = find(ts(:, 4) <= ts(1, 1) + 7200, 1, 'last') - 1;
    latestSet = setID(lastTrlWithinHr) - 1;
    
    % set up subsampling of J, R, CURBD over a fixed duration and number of trials (fixed across monkeys and recording days) 
    if latestSet <= setID(nTrls)
        setsToPlot = floor(linspace(2, latestSet, 20));
    else
        setsToPlot = floor(linspace(2, setID(nTrls), 20));
    end
    
    allTrlIDs = cell2mat(arrayfun(@(iSet) find(setID == iSet, minTrlsPerSet), setsToPlot, 'un', 0)');
    J_allsets = J_U_reordered(:, :, allTrlIDs);
    R_allsets_trunc_cell = arrayfun(@(iTrl) R_U_reordered{iTrl}(:, 1 : shortestTrl), allTrlIDs, 'un', 0);
    R_allsets_trunc_mat = cat(3, R_allsets_trunc_cell{:});
    
    % define histlims for J and R
    nBins = 50;
    JAllHistMin = min(sqrt(nUnits) * J_allsets(:));
    JAllHistMax = max(sqrt(nUnits) * J_allsets(:));
    JAllHistEdges = linspace(floor(JAllHistMin), ceil(JAllHistMax), nBins + 1);
    JAllHistCenters = JAllHistEdges(1 : end - 1) + (diff(JAllHistEdges) ./ 2); % get the centers. checked
    reshaped_J_allsets_mean = reshape(sqrt(nUnits) * mean(J_allsets, 3), nUnits .^ 2, 1);
    JAllHistCounts = histcounts(reshaped_J_allsets_mean, JAllHistEdges); % TO DO: look at it with hist and see
    RXLims = [-0.015 0.35]; % prctile(R_allsets_trunc_mat(:), [1 99]);
    RAllHistEdges = linspace(RXLims(1), RXLims(2),  nBins + 1);
    RAllHistCenters = RAllHistEdges(1 : end - 1) + (diff(RAllHistEdges) ./ 2);
    RAllHistCounts = histcounts(mean(R_allsets_trunc_mat, 3), RAllHistEdges);
    
    %% TRIAL-AVERAGED OVER ALL SUBSAMPLED TRIALS: J and R HISTS
    figure('color', 'w');
    set(gcf, 'units', 'normalized', 'outerposition', [0.2 0 0.4 1])
    subplot(2, 1, 1),
    semilogy(JAllHistCenters, JAllHistCounts ./ sum(JAllHistCounts, 2), 'o-', 'linewidth', 1.25, 'markersize', 5);
    set(gca, 'fontsize', fontSzM, 'fontweight', 'bold', 'ylim', [0 1], 'xlim', [-40 40]), 
    xlabel(gca, 'weight'), ylabel(gca, 'log density')
    title(['[', monkey, ssnDate, '] trial-averaged J over subsampled trials'], 'fontweight', 'bold')
    subplot(2, 1, 2),
    semilogy(RAllHistCenters, RAllHistCounts ./ sum(RAllHistCounts, 2), 'o-', 'linewidth', 1.25, 'markersize', 5);
    set(gca, 'fontsize', fontSzM, 'fontweight', 'bold', 'ylim', [0 1], 'xlim', RXLims), 
    xlabel(gca, 'activation'), ylabel(gca, 'log density')
    title(['[', monkey, ssnDate, '] trial-averaged R over subsampled trials'], 'fontweight', 'bold')
    print('-dtiff', '-r400', [histFigDir, 'J_and_act_avgd_', figNameStem])
    close
    
    %% set up CURBD avg over sets plot
    R_allsets_trunc = cell2mat(R_allsets_trunc_cell'); % (minTrlsPerSet * nSetsToPlot) x 1 array
    inds_allsets_trunc = [1; cumsum(repmat(shortestTrl, length(allTrlIDs), 1)) + 1]';
    [CURBD_allsets, CURBD_allsets_exc, CURBD_allsets_inh, ...
        avgCURBD_allsets, avgCURBD_allsets_exc, avgCURBD_allsets_inh, ...
        curbdRgns, nRegionsCURBD] = costaCURBD(J_allsets, R_allsets_trunc, inds_allsets_trunc, curbdRgns, minTrlsPerSet * length(setsToPlot));
    avgCURBD_resh = reshape(avgCURBD_allsets, nRegionsCURBD .^ 2, 1); % for getting YLims
    avgCURBD_mean = cell2mat(arrayfun(@(i) mean(avgCURBD_resh{i}, 1), 1 : nRegionsCURBD^2, 'un', 0)');
    CURBDYLims = 0.05; % max(abs(avgCURBD_mean(:))); % max(abs(prctile(avgCURBD_mean(:), [0.5 99.5])));
    
    %% TRIAL-AVERAGED MEAN CURRENT CURBD PLOT (ALL SETS)
    figure('color', 'w');
    set(gcf, 'units', 'normalized', 'outerposition', [0 0 1 1])
    count = 1;
    xTks = round(linspace(1, shortestTrl, 5)); % these will be trial averaged sample points
    axLPos = linspace(0.03, 0.85, nRegionsCURBD);
    axBPos = linspace(0.875, 0.06, nRegionsCURBD);
    axWPos = 0.095;
    axHPos = 0.095;
    
    for iTarget = 1 : nRegionsCURBD
        nUnitsTarget = numel(curbdRgns{iTarget, 2});
        
        for iSource = 1 : nRegionsCURBD
  
            subplot(nRegionsCURBD, nRegionsCURBD, count);
            hold all;
            count = count + 1;
            popMeanAvgCURBD = mean(avgCURBD_allsets{iTarget, iSource}, 1);
            patch([1 : shortestTrl, NaN], [popMeanAvgCURBD, NaN], [popMeanAvgCURBD, NaN], ...
                'linewidth', 0.75, 'EdgeColor', 'interp', 'Marker', '.', 'MarkerFaceColor', 'flat')
            axis tight, 
            line(gca, get(gca, 'xlim'), [0 0], 'linestyle', ':', 'linewidth', 0.75, 'color', [0.2 0.2 0.2])
            set(gca, 'ylim', [-1 * CURBDYLims, CURBDYLims], 'clim', [-1 * CURBDYLims, CURBDYLims], 'box', 'off', 'tickdir', 'out', 'fontsize', fontSzS)

            if iTarget == nRegionsCURBD && iSource == 1
                xlabel('time (s)', 'fontweight', 'bold');
                set(gca, 'xtick', xTks, 'xticklabel', num2str([(xTks * dtData) - dtData]', '%.1f'))
            else
                set(gca, 'xtick', '')
            end
            
            if iSource == 1
                ylbl = ylabel([curbdRgns{iTarget, 1}(1 : end - 3), '(', num2str(nUnitsTarget), ')'], 'fontweight', 'bold');
                ylbl.Position = [-44.5, 0, -1];
            end
            
            if iSource == 1 && iTarget == nRegionsCURBD
                set(gca, 'ytick', [-1 * CURBDYLims, 0, CURBDYLims])
                ytickangle(90)
            else
                set(gca, 'ytick', '')
            end

            set(gca, 'position', [axLPos(iSource), axBPos(iTarget), axWPos, axHPos])
            oldOPos = get(gca, 'OuterPosition');
            set(gca, 'OuterPosition', [1.01 * oldOPos(1), 1.01 * oldOPos(2), 0.99 * oldOPos(3), 0.99 * oldOPos(4)])
            title([curbdRgns{iSource, 1}(1 : end - 3), ' > ', curbdRgns{iTarget, 1}(1 : end - 3)], 'fontweight', 'bold');
        end
    end
    
    text(gca, -60, -1.5 * CURBDYLims, [monkey, ssnDate, ' CURBD (trial averaged)'], 'fontsize', fontSzM, 'fontweight', 'bold')
    cm = brewermap(250,'*RdBu');
    colormap(cm)
    print('-dtiff', '-r400', [CURBDFigDir, 'CURBD_avgd_', figNameStem])
    close
    
    %% TRIAL-AVERAGED MEAN CURRENT CURBD PLOT (ALL SETS, EXC)
    figure('color', 'w');
    set(gcf, 'units', 'normalized', 'outerposition', [0 0 1 1])
    count = 1;
    
    for iTarget = 1 : nRegionsCURBD
        nUnitsTarget = numel(curbdRgns{iTarget, 2});
        
        for iSource = 1 : nRegionsCURBD
  
            subplot(nRegionsCURBD, nRegionsCURBD, count);
            hold all;
            count = count + 1;
            
            popMeanAvgCURBD_exc = mean(avgCURBD_allsets_exc{iTarget, iSource}, 1);
            patch([1 : shortestTrl, NaN], [popMeanAvgCURBD_exc, NaN], [popMeanAvgCURBD_exc, NaN], ...
                'linewidth', 0.75, 'EdgeColor', 'interp', 'Marker', '.', 'MarkerFaceColor', 'flat')
            popMeanAvgCURBD_inh = mean(avgCURBD_allsets_inh{iTarget, iSource}, 1);
            patch([1 : shortestTrl, NaN], [popMeanAvgCURBD_inh, NaN], [popMeanAvgCURBD_inh, NaN], ...
                'linewidth', 0.75, 'EdgeColor', 'interp', 'Marker', '.', 'MarkerFaceColor', 'flat')
            axis tight,
            line(gca, get(gca, 'xlim'), [0 0], 'linestyle', ':', 'linewidth', 0.75, 'color', [0.2 0.2 0.2])
            set(gca, 'ylim', [-3 * CURBDYLims, 3 * CURBDYLims], 'clim', [-3 * CURBDYLims, 3 * CURBDYLims], 'box', 'off', 'tickdir', 'out', 'fontsize', fontSzS)
            
            if iTarget == nRegionsCURBD && iSource == 1
                xlabel('time (s)', 'fontweight', 'bold');
                set(gca, 'xtick', xTks, 'xticklabel', num2str([(xTks * dtData) - dtData]', '%.1f'))
            else
                set(gca, 'xtick', '')
            end
            
            if iSource == 1
                ylabel([curbdRgns{iTarget, 1}(1 : end - 3), '(', num2str(nUnitsTarget), ')'], 'fontweight', 'bold');
            end
            
            if iSource == 1 && iTarget == nRegionsCURBD
                set(gca, 'ytick', [-1 * CURBDYLims, 0, CURBDYLims])
            else
                set(gca, 'ytick', '')
            end
            
            set(gca, 'position', [axLPos(iSource), axBPos(iTarget), axWPos, axHPos])
            title([curbdRgns{iSource, 1}(1 : end - 3), ' > ', curbdRgns{iTarget, 1}(1 : end - 3)], 'fontweight', 'bold');
        end
    end
    
    text(gca, -60, -4 * CURBDYLims, [monkey, ssnDate, ' CURBD (trial averaged)'], 'fontsize', fontSzL, 'fontweight', 'bold')
    cm = brewermap(250,'*RdBu');
    colormap(cm)
    print('-dtiff', '-r400', [CURBDFigDir, 'CURBD_avgd_exc_inh_', figNameStem])
    close
    
    %% OVER SETS: set up histogram and CURBD plotting
    
    JHistCounts = NaN(length(setsToPlot), nBins);
    RHistCounts = NaN(length(setsToPlot), nBins);
    
    % colorscheme for plotting evolution within set
    overSsnColors = flipud(winter(length(setsToPlot))); % earlier is green, later blue% flipud(spring(length(setsToPlot))); % earlier is yellow, later is pink

    avgCURBD_set = cell(1, numel(setsToPlot));
    
    count2 = 0;
    for iSet = min(allSetIDs) : setID(nTrls)% min(allSetIDs) : max(allSetIDs) % setsToPlot 
        
        trlIDs = find(setID == iSet);
        fittedJ_samples = NaN([fittedConsJDim, numel(trlIDs)]);
        count = 1;
        
        % gotta load all J samples 
        for iTrl = 1 : length(trlIDs)
            mdlfnm = dir([mdlDir, monkey, ssnDate, filesep, 'rnn_', monkey, ssnDate, '_set*_trial', num2str(trlIDs(iTrl)), '.mat']);
            tmp = load(mdlfnm.name);
            mdl = tmp.RNN.mdl;
            fittedJ_samples(:, :, :, count) =  mdl.fittedConsJ(:, :, :);
            count = count + 1;
            clear mdl
        end
        
        R_set_dims = cell2mat(cellfun(@size, R_U_reordered(trlIDs), 'un', 0));
        inds_set = [1; cumsum(R_set_dims(:, 2)) + 1]';
        
        % reorder full J and R by region (includes intra and inter region)
        fullJ_samples = fittedJ_samples(newOrder, newOrder, :, :);
                
        % get R_set and J_set from loop in previous section
        J_set = J_U_reordered(:, :, trlIDs);
        R_set = cell2mat(R_U_reordered(trlIDs)');
        
       % get same trials from same sets, but trim down to shortest trial so you can trial average later... 
       R_set_trunc_cell = arrayfun(@(iTrl) R_U_reordered{iTrl}(:, 1 : shortestTrl), trlIDs, 'un', 0);
       R_set_trunc_mat = cat(3, R_set_trunc_cell{:}); % isequal(R_set_trunc_cell{3}, squeeze(R_set_trunc_mat(:, :, 3)))
       R_set_trunc = cell2mat(R_set_trunc_cell');
       inds_set_trunc = [1; cumsum(repmat(shortestTrl, length(trlIDs), 1)) + 1]';
       
        % initialize submatrices
        interJ_samples = fullJ_samples;
        intraJ_samples = zeros(size(fullJ_samples));
        
        % extract submatrices
        for iRgn = 1 : nRegions
            in_rgn = newRgnInds(iRgn) + 1 : newRgnInds(iRgn + 1);
            interJ_samples(in_rgn, in_rgn, :, :) = 0; % set intrargn values to zero
            intraJ_samples(in_rgn, in_rgn, :, :) = fullJ_samples(in_rgn, in_rgn, :, :); % pull only intrargn values
        end
        
        % grab vals of interest
        trlIDsCurrSet = allTrialIDs(trlIDs);
        activitySample = activitySampleAll(:, :, trlIDs);
        interJOverTrls(trlIDs) = squeeze(mean(interJ_samples, [1 2 3]));
        intraJOverTrls(trlIDs) = squeeze(mean(intraJ_samples, [1 2 3]));
        fullJOverTrls(trlIDs) = squeeze(mean(fullJ_samples, [1 2 3]));
        
        % want even distribution of sets over fixed dur for each subject/session
        if ismember(iSet, setsToPlot)
            count2 = count2 + 1;
            
            % CURBD that bitch
            [CURBD, CURBD_exc, CURBD_inh, avgCURBD, avgCURBD_exc, avgCURBD_inh, ~, ~] = costaCURBD(J_set, R_set_trunc, inds_set_trunc, curbdRgns, minTrlsPerSet);
            avgCURBD_set(count2) = {avgCURBD}; % collect into cell for later plotting of CURBD
            
            % get trial averaged truncated activity from minTrlsPerSet for
            % setsToPlot
            R_mean = mean(R_set_trunc_mat(:, :, 1 : minTrlsPerSet), 3);
            RHistCounts(count2, :) = histcounts(R_mean, RAllHistEdges);
            
            % plot of trial-averaged (within this set) fitted Js from the
            % last samplepoint in each trial
            % for J_source_to_target: reshape(sqrt(nSource) * J, nSource * nTarget, 1)
            J_mean = mean(J_set(:, :, 1 : minTrlsPerSet), 3);
            reshaped_J_mean = reshape(sqrt(nUnits) * J_mean, nUnits .^ 2, 1);
            [JHistCounts(count2, :)] = histcounts(reshaped_J_mean, JAllHistEdges); % TO DO: look at it with hist and see
        end
        
%         save([RNNSampleDir, 'classifier_matrices_IntraOnly', monkey, ssnDate, ...
%             '_set', num2str(iSet), '.mat'], 'activitySample', 'intraJ_samples', 'trlIDsCurrSet', 'iSet', '-v7.3')
%         
%         save([RNNSampleDir, 'classifier_matrices_InterOnly', monkey, ssnDate, ...
%             '_set', num2str(iSet), '.mat'], 'activitySample', 'interJ_samples', 'trlIDsCurrSet', 'iSet', '-v7.3')
%         
        clear interJSample intraJSample activitySample
    end
    
    %% TRIAL-AVERAGED (SPLIT BY SET) J HISTOGRAMS RE-FORMATTING (EACH SET)
    histFig = figure('color', 'w');
    set(gcf, 'units', 'normalized', 'outerposition', [0.2 0 0.4 1])
    
    if exist('colororder', 'file') ~= 0
        colororder(overSsnColors);
    end
    
    subplot(2, 1, 1),
    jHist = semilogy(JAllHistCenters, JHistCounts ./ sum(JHistCounts, 2), 'o-', 'linewidth', 1, 'markersize', 4);
    set(gca, 'fontsize', fontSzM, 'fontweight', 'bold', 'ylim', [0 1], 'xlim', [-40 40]), 
    xlabel(gca, 'weight'), ylabel(gca, 'log density')
    arrayfun(@(iLine) set(jHist(iLine), 'MarkerFaceColor', overSsnColors(iLine, :)), 1 : length(jHist));
    title(['[', monkey, ssnDate, '] trial-averaged J over sets'], 'fontweight', 'bold')
    legend(gca, cellstr([repmat('set ', length(setsToPlot), 1), num2str(setsToPlot')]), 'location', 'northeastoutside')
    subplot(2, 1, 2),
    rHist = semilogy(RAllHistCenters, RHistCounts ./ sum(RHistCounts, 2), 'o-', 'linewidth', 1, 'markersize', 4);
    set(gca, 'fontsize', fontSzM, 'fontweight', 'bold', 'ylim', [0 1], 'xlim', RXLims), 
    xlabel(gca, 'activation'), ylabel(gca, 'log density')
    arrayfun(@(iLine) set(rHist(iLine), 'MarkerFaceColor', overSsnColors(iLine, :)), 1 : length(rHist));
    title(['[', monkey, ssnDate, '] trial-averaged R over sets'], 'fontweight', 'bold')
    legend(gca, cellstr([repmat('set ', length(setsToPlot), 1), num2str(setsToPlot')]), 'location', 'northeastoutside')
    
    print('-dtiff', '-r400', [histFigDir, 'J_and_act_over_sets_', figNameStem])
    close
    
    %% CURBD OVER SETS
    
    figure('color', 'w');
    set(gcf, 'units', 'normalized', 'outerposition', [0 0 1 1])
    count = 1;
    xTks = round(linspace(1, shortestTrl, 5)); % these will be trial averaged sample points
    
    if exist('colororder', 'file') ~= 0
        colororder(overSsnColors);
    end
    
    for iTarget = 1 : nRegionsCURBD
        nUnitsTarget = numel(curbdRgns{iTarget, 2});
        
        for iSource = 1 : nRegionsCURBD
            
            subplot(nRegionsCURBD, nRegionsCURBD, count);
            hold all;
            count = count + 1;
            
            popMeanAvgCURBD = mean(avgCURBD_allsets{iTarget, iSource}, 1);
            popMeanSetCURBD = cell2mat(arrayfun(@(iSet) ...
                mean(avgCURBD_set{1, iSet}{iTarget, iSource}, 1), ...
                1 : length(setsToPlot), 'un', 0)');
            plot(1 : shortestTrl, popMeanSetCURBD, '-', 'linewidth', 0.75)
            axis tight,
            line(gca, get(gca, 'xlim'), [0 0], 'linestyle', ':', 'linewidth', 0.75, 'color', [0.2 0.2 0.2])
            set(gca, 'ylim', [-1 * CURBDYLims, CURBDYLims], 'clim', [-1 * CURBDYLims, CURBDYLims], 'box', 'off', 'tickdir', 'out', 'fontsize', fontSzS)
            
            if iTarget == nRegionsCURBD && iSource == 1
                xlabel('time (s)', 'fontweight', 'bold');
                set(gca, 'xtick', xTks, 'xticklabel', num2str([(xTks * dtData) - dtData]', '%.1f'))
            else
                set(gca, 'xtick', '')
            end
            
            if iSource == 1
                ylabel([curbdRgns{iTarget, 1}(1 : end - 3), '(', num2str(nUnitsTarget), ')'], 'fontweight', 'bold');
            end
            
            if iSource == 1 && iTarget == nRegionsCURBD
                set(gca, 'ytick', [-1 * CURBDYLims, 0, CURBDYLims])
            else
                set(gca, 'ytick', '')
            end
            
            set(gca, 'position', [axLPos(iSource), axBPos(iTarget), axWPos, axHPos]) % same resizing as CURBD_avg plot
            title([curbdRgns{iSource, 1}(1 : end - 3), ' > ', curbdRgns{iTarget, 1}(1 : end - 3)], 'fontweight', 'bold');
        end
    end
    
    text(gca, -60, -1.5 * CURBDYLims, [monkey, ssnDate, ' CURBD (over sets)'], 'fontsize', fontSzL, 'fontweight', 'bold')
    cm = brewermap(100,'*RdBu');
    colormap(cm)
    print('-dtiff', '-r400', [CURBDFigDir, 'CURBD_over_sets_', figNameStem])
    close
   
    %% check full, intra, inter
   
    JOverTrlsFigDir = [RNNfigdir, 'J_over_trls', filesep]; % pVarOverTrls, first trl convergence, last trl convergence
    if ~isfolder(JOverTrlsFigDir)
        mkdir(JOverTrlsFigDir)
    end
    
    trlsPerFig = 500;
    
    for f = 1 : ceil(nTrls / trlsPerFig) % new plot every five hundred trials
        figure('color', 'w')
        set(gcf, 'units', 'normalized', 'outerposition', [0.1 0.25 0.8 0.5])
        trlsToPlot = (f - 1) * trlsPerFig + 1 : (f * trlsPerFig);
        
        if trlsToPlot(end) > nTrls
            trlsToPlot = (f - 1) * trlsPerFig + 1 : nTrls;
        end
        
        plot(intraJOverTrls(trlsToPlot), 'b', 'linewidth', 1), hold on,
        plot(interJOverTrls(trlsToPlot), 'm', 'linewidth', 1)
        plot(fullJOverTrls(trlsToPlot), 'k', 'linewidth', 1)
        set(gca, 'fontweight', 'bold', 'fontsize', fontSzS, 'xlim', [1 trlsPerFig])
        set(gca, 'ylim', [-2.5e-3 2.5e-3], ...
            'xtick', floor(linspace(1, trlsPerFig, 6)), 'xticklabel', floor(linspace(trlsToPlot(1), trlsToPlot(1) + trlsPerFig, 6))) 
        xlabel('trial#', 'fontsize', fontSzM),
        legend(gca, {'intra', 'inter', 'full'}, 'location', 'northeast', 'autoupdate', 'off')
        line(gca, get(gca, 'xlim'), [0 0], 'linestyle', '-.', 'linewidth', 1, 'color', [0.2 0.2 0.2])
        title(gca, [monkey, ssnDate, ': mean consecutive J over trls'], 'fontsize', fontSzM, 'fontweight', 'bold')
        
        print('-dtiff', '-r400', [JOverTrlsFigDir, 'mean_J_over_trls_', figNameStem, '_', num2str(f)])
        close
    end
    
    %% pVar / chi2 for first and last J
    
    convergenceFigDir = [RNNfigdir, 'convergence', filesep]; % pVarOverTrls, first trl convergence, last trl convergence
    if ~isfolder(convergenceFigDir)
        mkdir(convergenceFigDir)
    end

    figure, set(gcf, 'units', 'normalized', 'outerposition', [0.1 0.15 0.8 0.7]),
    plot(pVarsTrls(trlSort), 'linewidth', 1.5, 'color', [0.05 0.4 0.15]), set(gca, 'ylim', [0.5 1], 'xlim', [1 nTrls]),
    title(gca, [monkey, ssnDate, ': pVar over trials'], 'fontsize', fontSzM, 'fontweight', 'bold')
    print('-dtiff', '-r400', [convergenceFigDir, 'pVar_over_trls_', figNameStem])
    close
    
    idx = 15; % randi(size(intraRgnJ, 1));
    figure('color', 'w')
    set(gcf, 'units', 'normalized', 'outerposition', [0.1 0.15 0.8 0.7])
    nRun = size(firstPVars, 2);
    subplot(2,4,1);
    hold on;
    imagesc(D_U{1}); colormap(jet), colorbar;
    axis tight; set(gca, 'clim', [0 1], 'xtick', [50:50:350], 'xticklabel', [50:50:350] * dtData)
    title('real'); xlabel('time (s)'), ylabel('units')
    set(gca,'Box','off','TickDir','out','FontSize', fontSzS);
    subplot(2,4,2);
    hold on;
    imagesc(R_U{1}); colormap(jet), colorbar;
    axis tight; set(gca, 'clim', [0 1], 'xtick', [50:50:350], 'xticklabel', [50:50:350] * dtData)
    title('model'); xlabel('time (s)'), ylabel('units')
    set(gca,'Box','off','TickDir','out','FontSize', fontSzS);
    subplot(2,4, [3 4 7 8]);
    hold all;
    plot(1 : size(R_U{1}, 2), R_U{1}(idx, :), 'linewidth', 1.5);
    plot(1 : size(D_U{1}, 2), D_U{1}(idx, :), 'linewidth', 1.5);
    axis tight; set(gca, 'ylim', [-0.05 0.5], 'xtick', [50:50:350], 'xticklabel', [50:50:350] * dtData)
    ylabel('activity');
    xlabel('time (s)'),
    lgd = legend('model','real','location','eastoutside');
    title('first trial')
    set(gca,'Box','off','TickDir','out','FontSize', fontSzS);
    subplot(2,4,5);
    hold on;
    plot(firstPVars(1:nRun), 'k', 'linewidth', 1.5);
    ylabel('pVar');  xlabel('run #'); set(gca, 'ylim', [-0.1 1.1], 'xlim', [1 nRun]); title(['final pVar=', num2str(firstPVars(end), '%.3f')])
    set(gca,'Box','off','TickDir','out','FontSize', fontSzS);
    subplot(2,4,6);
    hold on;
    plot(firstChi2(1:nRun), 'k', 'linewidth', 1.5)
    ylabel('chi2');  xlabel('run #'); set(gca, 'ylim', [-0.1 1.1], 'xlim', [1 nRun]); title(['final chi2=', num2str(firstChi2(end), '%.3f')])
    set(gca,'Box','off','TickDir','out','FontSize', fontSzS);
    print('-dtiff', '-r400', [convergenceFigDir, 'first_trl_convergence_', figNameStem])
    close
      
    % last
    figure('color', 'w')
    set(gcf, 'units', 'normalized', 'outerposition', [0.1 0.15 0.8 0.7])
    nRun = size(lastPVars, 2);
    subplot(2,4,1);
    hold on;
    imagesc(D_U{end}); colormap(jet), colorbar;
    axis tight; set(gca, 'clim', [0 1])
    title('real');
    set(gca,'Box','off','TickDir','out','FontSize', fontSzS);
    subplot(2,4,2);
    hold on;
    imagesc(R_U{end}); colormap(jet), colorbar;
    axis tight; set(gca, 'clim', [0 1])
    title('model');
    set(gca,'Box','off','TickDir','out','FontSize', fontSzS);
    subplot(2,4,[3 4 7 8]);
    hold all;
    plot(1 : size(R_U{end}, 2), R_U{end}(idx, :), 'linewidth', 1.5);
    plot(1 : size(D_U{end}, 2), D_U{end}(idx, :), 'linewidth', 1.5);
    axis tight; set(gca, 'ylim', [-0.05 0.5], 'xtick', [50:50:350], 'xticklabel', [50:50:350] * dtData)
    ylabel('activity');
    xlabel('time (s)'),
    lgd = legend('model','real','location','eastoutside');
    title(['last trial (', num2str(nTrls), ')'])
    set(gca,'Box','off','TickDir','out','FontSize', fontSzS);
    subplot(2,4,5);
    hold on;
    plot(lastPVars(1:nRun), 'k', 'linewidth', 1.5);
    ylabel('pVar');  xlabel('run #'); set(gca, 'ylim', [-0.1 1.1], 'xlim', [1 nRun]); title(['final pVar=', num2str(lastPVars(end), '%.3f')])
    set(gca,'Box','off','TickDir','out','FontSize', fontSzS);
    subplot(2,4,6);
    hold on;
    plot(lastChi2(1:nRun), 'k', 'linewidth', 1.5)
    ylabel('chi2'); xlabel('run #'); set(gca, 'ylim', [-0.1 1.1], 'xlim', [1 nRun]); title(['final chi2=', num2str(lastChi2(end), '%.3f')])
    set(gca,'Box','off','TickDir','out','FontSize', fontSzS);
    print('-dtiff', '-r400', [convergenceFigDir, 'last_trl_convergence_', figNameStem])
    close
end
   

    


