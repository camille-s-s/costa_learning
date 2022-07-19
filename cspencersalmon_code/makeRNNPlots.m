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

% in and outdirs
bd              = '~/Dropbox (BrAINY Crew)/costa_learning/';
mouseVer        = 'PINKY_VERSION/';
mdlDir          = [bd 'models/', mouseVer];
RNNfigdir       = [bd 'figures/', mouseVer];
RNNSampleDir    = [mdlDir 'model_samples/'];
spikeInfoPath   = [bd 'reformatted_data/'];

%%
addpath(genpath(bd))
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

for iFile = 1 % 1 : length(allFiles) % for each session....
    
    fName = allFiles(iFile).name;
    fID = fName(1:strfind(fName, '_') - 1);
    monkey = fID(1);
    ssnDate = fID(2:end);
    cd([mdlDir, monkey, ssnDate, filesep])
    currSsn = dir('rnn_*_set*_trial*.mat');
        
      allTrialIDs = unique(arrayfun(@(i) ...
        str2double(currSsn(i).name(strfind(currSsn(i).name,'trial') + 5 : end - 4)), 1:length(currSsn)));
    
    nTrls = length(allTrialIDs);
    
    % for first trial of a session, pull params (this field is empty for
    % all other trials in a session)
    firstTrl = find(arrayfun(@(i) isequal(currSsn(i).name, ['rnn_', monkey, ssnDate, '_set0_trial1.mat']), 1 : nTrls));
    spikeInfoName = [currSsn(firstTrl).name(5 : median(strfind(currSsn(firstTrl).name, '_'))-1), '_meta.mat'];
    load([spikeInfoPath, spikeInfoName], 'spikeInfo', 'trlInfo', 'dsAllEvents')
    load([mdlDir, monkey, ssnDate, filesep, currSsn(firstTrl).name]);
    binSize = RNN.mdl.dtData; % in sec
    tDataInit = RNN.mdl.tData;
    tRNNInit = RNN.mdl.tRNN;
 
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
    
    for i = 1 : nTrls
        
        mdlfnm = dir([mdlDir, monkey, ssnDate, filesep, 'rnn_', monkey, ssnDate, '_set*_trial', num2str(i), '.mat']);
        tmp = load(mdlfnm.name);
        mdl             = tmp.RNN.mdl;
        
        assert(mdl.dtRNN == 0.0005 & ...
            mdl. dtData == 0.01)
        
        iTarget         = mdl.iTarget;
        
        % turn a scrambly boy back into its OG order - target_unit_order(iTarget) = R_and_J_unit_order
        % aka targets and regions are unscrambled, R and J are scrambled

        [~, unscramble] = sort(iTarget); % Gives you indices from scrambled to unscrambled
        
        trlNum(i)           = mdl.iTrl;
        R_S{i}              = mdl.RMdlSample;
        J_S(:, :, i)        = mdl.J;
        J0_S(:, :, i)       = mdl.J0;
        D_S{i}              = mdl.targets(iTarget, :);
        D_U{i}              = mdl.targets;
        assert(isequal(mdl.targets, D_S{i}(unscramble, :))); % indices from scrambled back to original
        
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
    %% reorder full J and use it to get intra/inter only Js
        
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
    
    % for pulling trials from within a fixed time duration (here, hardcoded
    % to be 2 hours
    ts = trlInfo.event_times;
    lastTrlWithinHr = find(ts(:, 4) <= ts(1, 1) + 7200, 1, 'last') - 1;
    latestSet = setID(lastTrlWithinHr) - 1;
    
    % for JOverTrls subfigure showing development within sets over a fixed
    % duration of time
    setsToPlot = floor(linspace(2, latestSet, 5));
    
    % colorschemes for plotting evolution within set
    overSetColors = flipud(winter(minTrlsPerSet)); % earlier is green, later blue
    overSetColors_v2 = flipud(autumn(minTrlsPerSet)); % earlier is yellow, later red
    
    JTrajFig = figure('color', 'w');
    set(gcf, 'units', 'normalized', 'outerposition', [0 0.0275 1 0.95])
    
    
    nBins = 50;
    J_bincounts = NaN(length(setsToPlot), nBins);
    J_edgesnew = NaN(length(setsToPlot), nBins + 1);
    J_histcenters = NaN(length(setsToPlot), nBins);
    
    count2 = 0;
    
    for iSet = setsToPlot % min(allSetIDs) : max(allSetIDs)
        
        trlIDs = find(setID == iSet);
        fittedJ_samples = NaN([fittedConsJDim, numel(trlIDs)]);
        count = 1;
        
        % inds_sp = [];
        % R_set = [];
        % J_set = NaN(nUnits, nUnits, length(trlIDs));
        tData_set = [];
        tData_trlstart = NaN(1, length(trlIDs)); % [];
        % inds_set = [1, NaN(1, length(trlIDs) - 1)];
        
        % gotta load all J samples as well as tData
        for iTrl = 1 : length(trlIDs)
            mdlfnm = dir([mdlDir, monkey, ssnDate, filesep, 'rnn_', monkey, ssnDate, '_set*_trial', num2str(trlIDs(iTrl)), '.mat']);
            tmp = load(mdlfnm.name);
            mdl = tmp.RNN.mdl;
            fittedJ_samples(:, :, :, count) =  mdl.fittedConsJ(:, :, :);
            
            % for CURBD...
            % J_set(:, :, count) = mdl.J;
            % R_set = [R_set, mdl.RMdlSample];
            tData_set = [tData_set, mdl.tData];
            tData_trlstart(count) = mdl.tData(1); % tData_set(inds_trl) should be == to tData_trlstart
            
            % inds_sp = [inds_sp, mdl.sampleTimePoints + size(R_set, 2) - size(mdl.RMdlSample(newOrder, :), 2)];
            % inds_set(count + 1) = size(R_set, 2) + 1; % inds to start of each trial in R_set
            
            count = count + 1;
            clear mdl
        end
        
        R_set_dims = cell2mat(cellfun(@size, R_U_reordered(trlIDs), 'un', 0));
        inds_set = [1; cumsum(R_set_dims(:, 2)) + 1]';
        
        assert(isequal(tData_set(inds_set(1 : end - 1)), tData_trlstart))
        
        % reorder full J and R by region (includes intra and inter region)
        fullJ_samples = fittedJ_samples(newOrder, newOrder, :, :);
        % J_set = J_set(newOrder, newOrder, :);
        % R_set = R_set(newOrder, :);
        
        % get R_set and J_set from loop in previous section
        J_set = J_U_reordered(:, :, trlIDs);
        R_set = cell2mat(R_U_reordered(trlIDs)');
        
        % define histlims for J
        JHistMin = min(sqrt(nUnits) * J_U_reordered(:));
        JHistMax = max(sqrt(nUnits) * J_U_reordered(:));
        JHistEdges = linspace(floor(JHistMin), ceil(JHistMax), nBins + 1);
        JHistCenters = JHistEdges(1 : end - 1) + (diff(JHistEdges) ./ 2); % get the centers. checked

        % reshape samples for CURBD (sample)
        fullJ_sp_resh = reshape(fullJ_samples, nUnits, nUnits, nSamples * length(trlIDs));
        % isequal(fullJResh(:, :, nSamples + 1 : 2 * nSamples), fullJMatTmp(:, :, :, 2))
        
        % scratchpad goes here
        
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
        
        % want an even distribution of sets over a fixed duration of time
        % for each subject/session
        if ismember(iSet, setsToPlot)
            
            count2 = count2 + 1;
            
            % plot of trial-averaged (within this set) fitted Js from the
            % last samplepoint in each trial
            % for J_source_to_target: reshape(sqrt(nSource) * J, nSource * nTarget, 1)           
            J_mean = mean(J_set(:, :, 1 : minTrlsPerSet), 3);
            reshaped_J_mean = reshape(sqrt(nUnits) * J_mean, nUnits .^ 2, 1);
            [J_bincounts(count2, :)] = histcounts(reshaped_J_mean, JHistEdges); % TO DO: look at it with hist and see

            % plot mean-subtracted J evolution over samples for first minNTrls
            % of a set
            meanJsOverSet = squeeze(mean(fullJ_samples(:, :, :, 1 : minTrlsPerSet), [1 2])); % nSamples x nTrls
            
            % TOP ROW: plot of mean-subtracted (from start) J trajectory over first minNTrls of a set for five
            % equally spaced sets
            set(0, 'currentfigure', JTrajFig)
            subplot(3, length(setsToPlot), count2),
            jOverSetLines = plot(meanJsOverSet - meanJsOverSet(1, :), 'linewidth', 1.75, 'linestyle', '-'); 
            arrayfun(@(i) set(jOverSetLines(i), 'color', overSetColors(i, :)), 1 : minTrlsPerSet)
            set(gca, 'fontweight', 'bold', 'fontsize', 12, 'xlim', [0.5 nSamples + 0.5], 'xtick', 1 : 4 : nSamples, 'xticklabel', 1 : 4 : nSamples),
            xlabel('sample#')
            line(gca, get(gca, 'xlim'), [0 0], 'linestyle', ':', 'linewidth', 1, 'color', [0.2 0.2 0.2])
            title(['set ', num2str(iSet)], 'fontsize', 15)
            colAxes(1) = gca;
            tmpAxSz = get(colAxes(1), 'OuterPosition');
            set(colAxes(1), 'OuterPosition', [tmpAxSz(1), 0.9 * tmpAxSz(2), tmpAxSz(3), 1.18 * tmpAxSz(4)]) % H = 1.15
            if count2 == 3
                rTitle(3) = text(colAxes(1), 3, 1.15 * colAxes(1).Title.Position(2), 'MEAN-SUBTRACTED J', 'fontweight', 'bold', 'fontsize', 18);
            end
            if count2 == 1
                figTitle = text(colAxes(1), -1, 1.25 * colAxes(1).Title.Position(2), [monkey, ssnDate], 'fontweight', 'bold', 'fontsize', 20);
            end
            
            % MIDDLE ROW: plot of mean J over first minNTrls of a set for five equally
            % spaced sets, not mean subtracted
            subplot(3, length(setsToPlot), count2 + length(setsToPlot)),
            jLines = scatter(1 : minTrlsPerSet, mean(meanJsOverSet, 1), 75, overSetColors, 'filled', 'markeredgecolor', 'k'); grid minor; box on;
            set(gca, 'fontweight', 'bold', 'fontsize', 12, 'xlim', [0.5 minTrlsPerSet + 0.5], 'xtick', 1 : 8, 'xticklabel', 1 : 8), xlabel('trl#')
            jLinesSz = get(gca, 'OuterPosition');
            set(gca, 'OuterPosition', [jLinesSz(1:3) 0.7 * jLinesSz(4)])
            colAxes(2) = gca;
            if count2 == 3
                rTitle(2) = text(colAxes(2), 3, 1.15 * colAxes(2).Title.Position(2), 'RAW MEAN J', 'fontweight', 'bold', 'fontsize', 18);
            end
            
            % BOTTOM ROW: plot population mean activity evolution over samples
            meanActOverSet = squeeze(mean(activitySample(:, :, 1 : minTrlsPerSet), 1));
            subplot(3, length(setsToPlot), count2 + 2 * length(setsToPlot))
            actOverSetLines = plot(meanActOverSet, 'linewidth', 1.75, 'linestyle', '-');
            arrayfun(@(i) set(actOverSetLines(i), 'color', overSetColors_v2(i, :)), 1 : minTrlsPerSet)
            set(gca, 'fontweight', 'bold', 'fontsize', 12, 'xlim', [0.5 nSamples + 0.5], 'xtick', 1 : 4 : nSamples, 'xticklabel', 1 : 4 : nSamples),
            xlabel('sample#')
            colAxes(3) = gca;
            tmpAxSz = get(colAxes(3), 'OuterPosition');
            set(colAxes(3), 'OuterPosition', [tmpAxSz(1), 0.55 * tmpAxSz(2), tmpAxSz(3), 1.15 * tmpAxSz(4)]) % H = 1.1
            if count2 == 3
                rTitle(1) = text(colAxes(3), 0, 1.14 * colAxes(3).Title.Position(2), 'MEAN POPULATION ACTIVITY', 'fontweight', 'bold', 'fontsize', 18);
            end
            
            % re-size axes
            for axNum = 1 : 3
                axSz = get(colAxes(axNum), 'OuterPosition');
                switch count2
                    case 1 % more to the left and wider
                        set(colAxes(axNum), 'OuterPosition', [0.2 * axSz(1), axSz(2), 1.15 * axSz(3), axSz(4)])
                    otherwise % just wider
                        set(colAxes(axNum), 'OuterPosition', [axSz(1), axSz(2), 1.15 * axSz(3), axSz(4)])
                end
            end
        end
        
%         save([RNNSampleDir, 'classifier_matrices_IntraOnly', monkey, ssnDate, ...
%             '_set', num2str(iSet), '.mat'], 'activitySample', 'intraJ_samples', 'trlIDsCurrSet', 'iSet', '-v7.3')
%         
%         save([RNNSampleDir, 'classifier_matrices_InterOnly', monkey, ssnDate, ...
%             '_set', num2str(iSet), '.mat'], 'activitySample', 'interJ_samples', 'trlIDsCurrSet', 'iSet', '-v7.3')
%         
        clear interJSample intraJSample activitySample
    end
    
    
    %J_AND_ACT_OVER_SSN FORMATTING
    % re-size
    JTrajAxes = findall(JTrajFig, 'type', 'axes');
    newLeftPos = [repelem(0.81, 3), repelem(0.61, 3), repelem(0.41, 3), repelem(0.21, 3), repelem(0.01, 3)]';
    allLBWH = cell2mat(get(JTrajAxes, 'OuterPosition'));
    arrayfun(@(iAx) set(JTrajAxes(iAx), 'OuterPosition', [newLeftPos(iAx), allLBWH(iAx, 2 : end)]), 1 : length(JTrajAxes))
    
    % fix ylims by row
    iRowMultip = [0.06, 0.2, 0.18];
    iRowYLims = [0.02 0.06; -5e-4 20e-4; -1e-6 1e-6];
    
    for iRow = 1 : 3
        tmpYLims = cell2mat(arrayfun(@(iAx) get(JTrajAxes(iAx), 'Ylim'), iRow : 3 : length(JTrajAxes), 'un', 0)');
        YMinRow = iRowYLims(iRow, 1); YMaxRow = iRowYLims(iRow, 2);
        arrayfun(@(iAx) set(JTrajAxes(iAx), 'YLim', [YMinRow, YMaxRow]), iRow : 3 : length(JTrajAxes))
        
        % re-position titles
        oldRowTitlePos = get(rTitle(iRow), 'Position');
        newRowTitleYPos = iRowMultip(iRow) * (abs(YMinRow) + abs(YMaxRow)) + YMaxRow;
        set(rTitle(iRow), 'Position', [oldRowTitlePos(1), newRowTitleYPos, 0])
        
        if iRow == 3
            oldFigTitlePos = get(figTitle, 'Position');
            set(figTitle, 'Position', [oldFigTitlePos(1), newRowTitleYPos, 0])
        end
    end
    
    if ~isfolder([RNNfigdir, 'J_and_act_over_ssn', filesep])
        mkdir([RNNfigdir, 'J_and_act_over_ssn', filesep])
    end
    
    print('-dtiff', '-r400', [RNNfigdir, 'J_and_act_over_ssn', filesep, monkey, ssnDate, '_J_and_act_over_ssn'])
    close
    
    % TRIAL-AVERAGED J HISTOGRAMS RE-FORMATTING
    overSsnColors = flipud(spring(length(setsToPlot))); % earlier is yellow, later is pink
    histFig = figure('color', 'w');
    set(gcf, 'units', 'normalized', 'outerposition', [0 0.0275 1 0.95])
        if exist('colororder','file') ~= 0
        colororder(overSsnColors);
    end
    
    
    subplot(2, 1, 1),
    
    % normalizing each by area under curve
    jHist = semilogy(JHistCenters, J_bincounts ./ sum(J_bincounts, 2), 'o-', 'linewidth', 0.8, 'markersize', 1.5);
    set(gca, 'fontsize', 6, 'ylim', [0 1], 'xlim', hist_xlims)
    
    %% check full, intra, inter
    
    if ~isfolder([RNNfigdir, 'JOVerTrls', filesep])
        mkdir([RNNfigdir, 'JOverTrls', filesep])
    end
    
    trlsPerFig = 500;
    
    for f = 1 : ceil(nTrls / trlsPerFig) % new plot every five hundred trials
        figure('color', 'w')
        set(gcf, 'units', 'normalized', 'outerposition', [0.1 0.25 0.8 0.5])
        trlsToPlot = (f - 1) * trlsPerFig + 1 : (f * trlsPerFig);
        
        if trlsToPlot(end) > nTrls
            trlsToPlot = (f - 1) * trlsPerFig + 1 : nTrls;
        end
        
        plot(intraJOverTrls(trlsToPlot), 'b:', 'linewidth', 1.75), hold on,
        plot(interJOverTrls(trlsToPlot), 'm:', 'linewidth', 1.75)
        plot(fullJOverTrls(trlsToPlot), 'k', 'linewidth', 1.75)
        set(gca, 'fontweight', 'bold', 'xlim', [1 trlsPerFig])
        set(gca, 'ylim', [-2.5e-3 2.5e-3], ...
            'xtick', floor(linspace(1, trlsPerFig, 6)), 'xticklabel', floor(linspace(trlsToPlot(1), trlsToPlot(1) + trlsPerFig, 6))) 
        xlabel('trial#', 'fontsize', 14),
        legend(gca, {'intra', 'inter', 'full'}, 'location', 'northeast', 'autoupdate', 'off')
        
        line(gca, get(gca, 'xlim'), [0 0], 'linestyle', '-.', 'linewidth', 1, 'color', [0.2 0.2 0.2])
        title(gca, [monkey, ssnDate, ': mean consecutive J over trls'], 'fontsize', 14, 'fontweight', 'bold')
        
        print('-dtiff', '-r400', [RNNfigdir, 'JOverTrls', filesep, monkey, ssnDate, '_intraVsIntervsFull_', num2str(f)])
        close
    end
    
    %% pVar / chi2 for first and last J
    
    if ~isfolder([RNNfigdir, 'pVar', filesep])
        mkdir([RNNfigdir, 'pVar', filesep])
    end
    
    figure, set(gcf, 'units', 'normalized', 'outerposition', [0.1 0.15 0.8 0.7]),
    plot(pVarsTrls(trlSort), 'linewidth', 1.5, 'color', [0.05 0.4 0.15]), set(gca, 'ylim', [0.1 1], 'xlim', [1 nTrls]),
    title(gca, [monkey, ssnDate, ': pVar over trials'], 'fontsize', 14, 'fontweight', 'bold')
    print('-dtiff', '-r400', [RNNfigdir, 'pVar', filesep, monkey, ssnDate, '_eachTrl'])
    close
    
    if ~isfolder([RNNfigdir, 'convergence', filesep])
        mkdir([RNNfigdir, 'convergence', filesep])
    end
    
    idx = 15; % randi(size(intraRgnJ, 1));
    figure('color', 'w')
    set(gcf, 'units', 'normalized', 'outerposition', [0.1 0.15 0.8 0.7])
    nRun = size(firstPVars, 2);
    subplot(2,4,1);
    hold on;
    imagesc(D_U{1}); colormap(jet), colorbar;
    axis tight; set(gca, 'clim', [0 1], 'xtick', [50:50:350], 'xticklabel', [50:50:350] * dtData)
    title('real'); xlabel('time (s)'), ylabel('units')
    set(gca,'Box','off','TickDir','out','FontSize',14);
    subplot(2,4,2);
    hold on;
    imagesc(R_U{1}); colormap(jet), colorbar;
    axis tight; set(gca, 'clim', [0 1], 'xtick', [50:50:350], 'xticklabel', [50:50:350] * dtData)
    title('model'); xlabel('time (s)'), ylabel('units')
    set(gca,'Box','off','TickDir','out','FontSize',14);
    subplot(2,4, [3 4 7 8]);
    hold all;
    plot(1 : size(R_U{1}, 2), R_U{1}(idx, :), 'linewidth', 1.5);
    plot(1 : size(D_U{1}, 2), D_U{1}(idx, :), 'linewidth', 1.5);
    axis tight; set(gca, 'ylim', [-0.05 0.5], 'xtick', [50:50:350], 'xticklabel', [50:50:350] * dtData)
    ylabel('activity');
    xlabel('time (s)'),
    lgd = legend('model','real','location','eastoutside');
    title('first trial')
    set(gca,'Box','off','TickDir','out','FontSize',14);
    subplot(2,4,5);
    hold on;
    plot(firstPVars(1:nRun), 'k', 'linewidth', 1.5);
    ylabel('pVar');  xlabel('run #'); set(gca, 'ylim', [-0.1 1]); title(['final pVar=', num2str(firstPVars(end), '%.3f')])
    set(gca,'Box','off','TickDir','out','FontSize',14);
    subplot(2,4,6);
    hold on;
    plot(firstChi2(1:nRun), 'k', 'linewidth', 1.5)
    ylabel('chi2');  xlabel('run #'); set(gca, 'ylim', [-0.1 1]); title(['final chi2=', num2str(firstChi2(end), '%.3f')])
    set(gca,'Box','off','TickDir','out','FontSize',14);
    saveas(gcf, [RNNfigdir, 'convergence', filesep, monkey, ssnDate, '_firstTrlConvergence'], 'svg')
    print('-dtiff', '-r400', [RNNfigdir, 'convergence', filesep, monkey, ssnDate, '_firstTrlConvergence'])
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
    set(gca,'Box','off','TickDir','out','FontSize',14);
    subplot(2,4,2);
    hold on;
    imagesc(R_U{end}); colormap(jet), colorbar;
    axis tight; set(gca, 'clim', [0 1])
    title('model');
    set(gca,'Box','off','TickDir','out','FontSize',14);
    subplot(2,4,[3 4 7 8]);
    hold all;
    plot(1 : size(R_U{end}, 2), R_U{end}(idx, :), 'linewidth', 1.5);
    plot(1 : size(D_U{end}, 2), D_U{end}(idx, :), 'linewidth', 1.5);
    axis tight; set(gca, 'ylim', [-0.1 1])
    ylabel('activity');
    xlabel('time (s)'),
    lgd = legend('model','real','location','eastoutside');
    title('last trial')
    set(gca,'Box','off','TickDir','out','FontSize',14);
    subplot(2,4,5);
    hold on;
    plot(lastPVars(1:nRun), 'k', 'linewidth', 1.5);
    ylabel('pVar');  xlabel('run #'); set(gca, 'ylim', [-0.1 1]); title(['final pVar=', num2str(lastPVars(end), '%.3f')])
    set(gca,'Box','off','TickDir','out','FontSize',14);
    subplot(2,4,6);
    hold on;
    plot(lastChi2(1:nRun), 'k', 'linewidth', 1.5)
    ylabel('chi2'); xlabel('run #'); set(gca, 'ylim', [-0.1 1]); title(['final chi2=', num2str(lastChi2(end), '%.3f')])
    set(gca,'Box','off','TickDir','out','FontSize',14);
    print('-dtiff', '-r400', [RNNfigdir, 'convergence', filesep, monkey, ssnDate, '_lastTrlConvergence'])
    close

    %% vectorized J: inter, intra TO DO 2022/01/11: CHECK IF THIS WORKS 
    
%     intraJResh = NaN(sum(nUnitsAll)^2, nSamples, nTrls);
%     interJResh = NaN(sum(nUnitsAll)^2, nSamples, nTrls);
%     fullJResh = NaN(sum(nUnitsAll)^2, nSamples, nTrls);
%     
    % self proof
%     tmp = intraJMat(:, :, 1, 1);
%     tmpResh = reshape(tmp, 1, numel(tmp));
%     isequal(tmp, reshape(tmpResh, size(tmp)))
    
%     for iSet = min(allSetIDs) : max(allSetIDs)
%         
%         trlIDs = find(setID == iSet);
%         
%             intraJResh = NaN(sum(nUnitsAll)^2, nSamples, numel(trlIDs));
%     interJResh = NaN(sum(nUnitsAll)^2, nSamples, numel(trlIDs));
%     fullJResh = NaN(sum(nUnitsAll)^2, nSamples, numel(trlIDs));
%     
%         
%         for iTrl = 1 : length(trlIDs)
%             
%             intraJResh(:, :, iTrl) = cell2mat(arrayfun(@(iSample) ...
%                 reshape(intraJMat(:, :, iSample, trlIDs(iTrl)), 1, sum(nUnitsAll)^2), 1 : nSamples, 'un', 0)')';
%           
%             interJResh(:, :, iTrl) = cell2mat(arrayfun(@(iSample) ...
%                 reshape(interJMat(:, :, iSample, trlIDs(iTrl)), 1, sum(nUnitsAll)^2), 1 : nSamples, 'un', 0)')';
%           
%             fullJResh(:, :, iTrl) = cell2mat(arrayfun(@(iSample) ...
%                 reshape(fullJMat(:, :, iSample, trlIDs(iTrl)), 1, sum(nUnitsAll)^2), 1 : nSamples, 'un', 0)')';
%             
% %             intraJResh{iTrl} = arrayfun(@(iSample) ...
% %                 reshape(squeeze(intraJ{iTrl}(:, :, iSample)), 1, size(in_rgn, 1)^2), 1 : nSamples, 'un', false);
% %             interJResh{iTrl} = arrayfun(@(iSample) ...
% %                 reshape(squeeze(interJ{iTrl}(:, :, iSample)), 1, sum(nUnitsAll)^2), 1 : nSamples, 'un', false);
% %             fullJResh{iTrl} = arrayfun(@(iSample) ...
% %                 reshape(squeeze(fullJ{iTrl}(:, :, iSample)), 1, sum(nUnitsAll)^2), 1 : nSamples, 'un', false);
%         end
%         
%         intraJSampleResh = intraJResh(:, :, trlIDs);
%         interJSampleResh = interJResh(:, :, trlIDs);
%         trlIDsCurrSet = allTrialIDs(trlIDs);
%         activitySample = activitySampleAll(:, :, trlIDs);
% 
%         save([RNNSampleDir, 'classifier_matrices_resh_IntraOnly', monkey, ssnDate, ...
%             '_set', num2str(iSet), '.mat'], 'activitySample', 'intraJSampleResh', 'trlIDsCurrSet', 'iSet', '-v7.3')
%         save([RNNSampleDir, 'classifier_matrices_resh_InterOnly', monkey, ssnDate, ...
%             '_set', num2str(iSet), '.mat'], 'activitySample', 'interJSampleResh', 'trlIDsCurrSet', 'iSet', '-v7.3')
%         
%     end

    %%
    
%     for iTrl = 1 : nTrls
%         fullJResh{iTrl} = arrayfun(@(iSample) ...
%                 reshape(squeeze(fullJ{iTrl}(:, :, iSample)), 1, sum(nUnitsAll)^2), 1 : nSamples, 'un', false);
%         interJResh{iTrl} = arrayfun(@(iSample) ...
%                 reshape(squeeze(interJ{iTrl}(:, :, iSample)), 1, sum(nUnitsAll)^2), 1 : nSamples, 'un', false);
%     end
%     
     %% PCA OF TRAINED MODEL ACTIVITY MATRIX
        
    % remove bad units from  activity
    d = cell2mat(R_U');
    [~, FR_cutoff] = isoutlier(mean(d, 2), 'percentile', [0.5 100]); % [0.5 100] % throw out outliers since not good for PCA
    bad_units = mean(d, 2) <= FR_cutoff;
    
    % throw out any trials that are weirdly long or short
    trlDursRNN = arrayfun(@(s) size(R_U{s}, 2), 1 : size(R_U, 1));
    [~, minTrlLen, maxTrlLen] = isoutlier(trlDursRNN, 'percentile', [5 95]); % [1 99]
    
    if maxTrlLen > median(trlDursRNN) + (2 * std(trlDursRNN))
        maxTrlLen = median(trlDursRNN) + (2 * std(trlDursRNN));
    end
    
    bad_trls = trlDursRNN >= maxTrlLen | trlDursRNN <= minTrlLen;
    Rtmp = R_U(~bad_trls);
    durBtwnTrls = [NaN; diff(trlInfo.event_times(:, 1))];
    T = trlInfo(~bad_trls, :);
    durBtwnTrls = durBtwnTrls(~bad_trls, :);
    
    % elapsed time between events (1 = fixation, 2 = stim, 3 = choice, 4 =
    % outcome, 5 = time of next trl fixation)
    avgEventTS = [mean(T.aligned_event_times, 1), nanmean(durBtwnTrls)];
    eventInds = round(avgEventTS / dtData);
    eventInds(1) = 1;
    
    % pad remaining units and trials to same length
    longestTrl = max(arrayfun(@(s) max(size(Rtmp{s}, 2)), 1 : size(Rtmp, 1)));
    trlsPadded = arrayfun(@(s) ...
        [Rtmp{s}, NaN(size(Rtmp{s}, 1), longestTrl - size(Rtmp{s}, 2))], 1 : size(Rtmp, 1), 'un', false);
    X = cell2mat(trlsPadded);
    
    % trial average them
    trlsAvgR = cell2mat(arrayfun( @(n) nanmean(reshape(X(n, :), longestTrl, (nTrls - sum(bad_trls))), 2), 1 : size(X, 1), 'un', false))';
    trlsAvgR(bad_units, :) = [];
    
    % coeffs from all activity, trial averaged and padded
    [wR, scores, eigen, ~, pvar, mu] = pca(trlsAvgR', 'Algorithm', 'svd', 'Centered', center_data, 'Economy', 0);
    
    % estimate effective dimensionality
    if ~isfolder([RNNfigdir, 'effDimMdlActivity', filesep])
        mkdir([RNNfigdir, 'effDimMdlActivity', filesep])
    end
    
    figure('color', 'w'), plot(1 : length(eigen), cumsum(pvar), 'linewidth', 1.5, 'color', 'b')
    set(gca, 'fontweight', 'bold', 'fontsize', 13, 'ylim', [0 110], 'xlim', [1 round(0.1 * length(eigen))])
    grid minor, xlabel('# PCs (10% of actual total)'), ylabel('% variance explained')
    nPC_cutoff = find(cumsum(pvar) >= 99,1, 'first');
    line(gca, [nPC_cutoff nPC_cutoff], get(gca, 'ylim'), 'linestyle', '--', 'linewidth', 1.5, 'color', 'k')
    text(gca, round(1.2 * nPC_cutoff), 10, ['#PCs = ', num2str(nPC_cutoff)], 'fontweight', 'bold')
    
    title(gca, [monkey, ssnDate, ': eff dim (99% var exp) for avg mdl activity (#', num2str(nTrls - sum(bad_trls)), ' trls)'], 'fontsize', 14, 'fontweight', 'bold')
    print('-dtiff', '-r400', [RNNfigdir, 'effDimMdlActivity', filesep, monkey, ssnDate, '_effDimMdlActivity'])
    close
    
    % project data into low D space
    if ~isfolder([RNNfigdir, 'topPCsMdlActivity', filesep])
        mkdir([RNNfigdir, 'topPCsMdlActivity', filesep])
    end
    
    figure('color', 'w'); hold on
    set(gcf, 'units', 'normalized', 'outerposition', [0.25 0.1 0.5 0.8])
    cm = colormap(cool);
    colormap(cm)
    
    % project padded trial averaged activity (all) into low D space
    projData = trlsAvgR';
    tempProj = (projData - repmat(mu,size(projData,1),1)) * wR;
    x = tempProj;
    plot3(x(:, 1), x(:, 2), x(:, 3), 'color', [0.66 0.66 0.66], 'linewidth', 2)
    plot3(x(10 : 10 : end - 10, 1), x(10 : 10 : end - 10, 2), x(10 : 10 : end - 10, 3), 'linestyle', 'none', 'marker', 'o', 'markersize', 8, 'markeredgecolor', [0 0 0.2], 'markerfacecolor', 'w')
    
    % mark events on plot
    eventColors = {'g', 'c', 'b', 'm', 'g'};
    eventName = {'fixation (current trial)', 'stimulus', 'choice', 'outcome', 'fixation (next trial)'};
    
    for event = 1 : length(eventInds)
        eventLbls(event) = plot3(x(eventInds(event), 1), x(eventInds(event), 2), x(eventInds(event), 3), ...
            'linestyle', 'none', 'marker', 'd', 'markersize', 15, 'markeredgecolor', 'k', 'markerfacecolor', eventColors{event});
    end
    
    set(eventLbls(end), 'marker', 's') % mark next trial w different shape
    legend(gca, eventLbls, eventName)
    
    set(gca, 'fontweight', 'bold', 'fontsize', 13)
    xlabel('PC 1'), ylabel('PC 2'), zlabel('PC 3')
    grid minor
    view(3)
    title(gca, [monkey, ssnDate, ': avg mdl activity projected into top 3 PCs'], 'fontsize', 14, 'fontweight', 'bold')
    saveas(gcf, [RNNfigdir, 'topPCsMdlActivity', filesep, monkey, ssnDate, '_topPCsMdlActivity'], 'svg')
    print('-dtiff', '-r400', [RNNfigdir, 'topPCsMdlActivity', filesep, monkey, ssnDate, '_topPCsMdlActivity'])
    close
    
    %% WIP: PCA of activity over nD trials from chosen set idk look up jPCA trajectories look up difference between
    
    % remove bad units from  activity
    d = cell2mat(D_U');
    [~, FR_cutoff] = isoutlier(mean(d, 2), 'percentile', [0.5 100]); % [0.5 100] % throw out outliers since not good for PCA
    bad_units = mean(d, 2) <= FR_cutoff;
    
    % throw out any trials that are weirdly long or short
    trlDursRNN = arrayfun(@(s) size(D_U{s}, 2), 1 : size(D_U, 1));
    [~, minTrlLen, maxTrlLen] = isoutlier(trlDursRNN, 'percentile', [5 95]); % [1 99]
    
    if maxTrlLen > median(trlDursRNN) + (2 * std(trlDursRNN))
        maxTrlLen = median(trlDursRNN) + (2 * std(trlDursRNN));
    end
    
    bad_trls = trlDursRNN >= maxTrlLen | trlDursRNN <= minTrlLen;
    Dtmp = D_U(~bad_trls);
    durBtwnTrls = [NaN; diff(trlInfo.event_times(:, 1))];
    T = trlInfo(~bad_trls, :);
    durBtwnTrls = durBtwnTrls(~bad_trls, :);
    
    % elapsed time between events (1 = fixation, 2 = stim, 3 = choice, 4 =
    % outcome, 5 = time of next trl fixation)
    avgEventTS = [mean(T.aligned_event_times, 1), nanmean(durBtwnTrls)];
    eventInds = round(avgEventTS / dtData);
    eventInds(1) = 1;
    
    % pad remaining units and trials to same length
    longestTrl = max(arrayfun(@(s) max(size(Dtmp{s}, 2)), 1 : size(Dtmp, 1)));
    trlsPadded = arrayfun(@(s) ...
        [Dtmp{s}, NaN(size(Dtmp{s}, 1), longestTrl - size(Dtmp{s}, 2))], 1 : size(Dtmp, 1), 'un', false);
    X = cell2mat(trlsPadded);
    
    % trial average them
    trlsAvgD = cell2mat(arrayfun( @(n) nanmean(reshape(X(n, :), longestTrl, (nTrls - sum(bad_trls))), 2), 1 : size(X, 1), 'un', false))';
    trlsAvgD(bad_units, :) = [];
    
    % coeffs from all activity, trial averaged and padded
    [w, scores, eigen, ~, pvar, mu] = pca(trlsAvgD', 'Algorithm', 'svd', 'Centered', center_data, 'Economy', 0);
    
    % estimate effective dimensionality
    if ~isfolder([RNNfigdir, 'effDimAllActivity', filesep])
        mkdir([RNNfigdir, 'effDimAllActivity', filesep])
    end
    
    figure('color', 'w'), plot(1 : length(eigen), cumsum(pvar), 'linewidth', 1.5, 'color', 'b')
    set(gca, 'fontweight', 'bold', 'fontsize', 13, 'ylim', [0 110], 'xlim', [1 round(0.1 * length(eigen))])
    grid minor, xlabel('# PCs (10% of actual total)'), ylabel('% variance explained')
    nPC_cutoff = find(cumsum(pvar) >= 99,1, 'first');
    line(gca, [nPC_cutoff nPC_cutoff], get(gca, 'ylim'), 'linestyle', '--', 'linewidth', 1.5, 'color', 'k')
    text(gca, round(1.2 * nPC_cutoff), 10, ['#PCs = ', num2str(nPC_cutoff)], 'fontweight', 'bold')
    
    title(gca, [monkey, ssnDate, ': eff dim (99% var exp) for avg activity (#', num2str(nTrls - sum(bad_trls)), ' trls)'], 'fontsize', 14, 'fontweight', 'bold')
    print('-dtiff', '-r400', [RNNfigdir, 'effDimAllActivity', filesep, monkey, ssnDate, '_effDimAllActivity'])
    close
    
    % project data into low D space
    if ~isfolder([RNNfigdir, 'topPCsAllActivity', filesep])
        mkdir([RNNfigdir, 'topPCsAllActivity', filesep])
    end
    
    figure('color', 'w'); hold on
    set(gcf, 'units', 'normalized', 'outerposition', [0.25 0.1 0.5 0.8])
    cm = colormap(cool);
    colormap(cm)
    
    % project padded trial averaged activity (all) into low D space
    projData = trlsAvgD';
    tempProj = (projData - repmat(mu,size(projData,1),1)) * wR;
    x = tempProj;
    plot3(x(:, 1), x(:, 2), x(:, 3), 'color', 'k', 'linewidth', 2)
    plot3(x(10 : 10 : end - 10, 1), x(10 : 10 : end - 10, 2), x(10 : 10 : end - 10, 3), 'linestyle', 'none', 'marker', 'o', 'markersize', 8, 'markeredgecolor', 'k', 'markerfacecolor', 'w')
    
    % mark events on plot
    eventColors = {'g', 'c', 'b', 'm', 'g'};
    eventName = {'fixation (current trial)', 'stimulus', 'choice', 'outcome', 'fixation (next trial)'};
    
    for event = 1 : length(eventInds)
        eventLbls(event) = plot3(x(eventInds(event), 1), x(eventInds(event), 2), x(eventInds(event), 3), ...
            'linestyle', 'none', 'marker', 'd', 'markersize', 15, 'markeredgecolor', 'k', 'markerfacecolor', eventColors{event});
    end
    
    set(eventLbls(end), 'marker', 's') % mark next trial w different shape
    legend(gca, eventLbls, eventName)
    
    set(gca, 'fontweight', 'bold', 'fontsize', 13)
    xlabel('PC 1'), ylabel('PC 2'), zlabel('PC 3')
    grid minor
    view(3)
    title(gca, [monkey, ssnDate, ': avg real activity projected into top 3 PCs'], 'fontsize', 14, 'fontweight', 'bold')
    saveas(gcf, [RNNfigdir, 'topPCsAllActivity', filesep, monkey, ssnDate, '_realActPjnIntoMdlPCSpace'], 'svg')
    print('-dtiff', '-r400', [RNNfigdir, 'topPCsAllActivity', filesep, monkey, ssnDate, '_topPCsAllActivity'])
    close
    
   
     %% 2021-10-27 : EXTRACT J SAMPLES FOR CLASSIFIER
     
%      activitySampleMat = NaN(sum(nUnitsAll), nSamples, nTrls);
%      
%      % Sloppy grabbing
%      for iTrl = 1 : nTrls
%          sampleTimePoints = allSPTimePoints(iTrl, :);
%          % to save! note that J is reordered by region
%          activitySampleAll{iTrl} = R_U{iTrl}(:, sampleTimePoints); % TO DO: FIGURE OUT WHAT THE ISSUE IS W INDEXING FROM MIN_LEN - SHOULDNT NEED TO TRANSFORM IT FOR PULLING MODEL ACTIVITY
%          activitySampleMat(:, :, iTrl) = activitySampleAll{iTrl};
%      end
     
     meanIntraJAcrossTrls = mean(squeeze(cell2mat(arrayfun(@(iTrl) mean(intraJ{iTrl}, [1 2]), 1 : nTrls, 'un', false))), 1);
     meanActAcrossTrls = mean(cell2mat(arrayfun(@(iTrl) mean(activitySampleAll{iTrl}, 1), 1 : nTrls, 'un', false)'), 1);
     stdIntraJAcrossTrls = std(squeeze(cell2mat(arrayfun(@(iTrl) mean(intraJ{iTrl}, [1 2]), 1 : nTrls, 'un', false))), 0, 1) ./ sqrt(sum(nUnitsAll));
     stdActAcrossTrls = std(cell2mat(arrayfun(@(iTrl) mean(activitySampleAll{iTrl}, 1), 1 : nTrls, 'un', false)'), 0, 1) ./ sqrt(sum(nUnitsAll));
     
     figure, errorbar(meanIntraJAcrossTrls, stdIntraJAcrossTrls, 'linewidth', 1.5), set(gca, 'fontweight', 'bold', 'fontsize', 12, 'xlim', [0.5 nSamples + 0.5]), xlabel('sample point#'), ylabel('pop mean J')
     saveas(gcf, [RNNfigdir, 'intraRgnDeltaJs_consecutiveSamplesSingleTrl', filesep, monkey, ssnDate, '_meanJAcrossTrls'], 'svg')
     print(gcf, '-dtiff', '-r400', [RNNfigdir, 'intraRgnDeltaJs_consecutiveSamplesSingleTrl', filesep, monkey, ssnDate, '_meanJAcrossTrls']),
     close
     
     figure, errorbar(meanActAcrossTrls, stdActAcrossTrls, 'linewidth', 1.5), set(gca, 'fontweight', 'bold', 'fontsize', 12, 'xlim', [0.5 nSamples + 0.5]), xlabel('sample point#'), ylabel('pop mean FR')
     print(gcf, '-dtiff', '-r400', [RNNfigdir, 'intraRgnDeltaJs_consecutiveSamplesSingleTrl', filesep, monkey, ssnDate, '_meanActAcrossTrls']),
     close
     
%      for iSet = min(allSetIDs) : max(allSetIDs)
%          activitySample = activitySampleAll(setID == iSet);
%          intraJSample = intraRgnConsJ(setID == iSet);
%          intraJSampleResh = intraRgnConsJResh(setID == iSet);
%          interJSample = interRgnConsJ(setID == iSet);
%          interJSampleResh = interRgnConsJResh(setID == iSet);
%          
%          trlIDsCurrSet = allTrialIDs(setID == iSet);
%          
%          % for the last trial of the first and last included sets, take the mean over all
%          % samples to plot representative histograms
%          if iSet == min(allSetIDs)
%              meanJOverLastTrlFirstSet = mean(intraJSample{end}, 3);
%              meanJOverLastTrlFirstSetPlot = reshape(sqrt(sum(nUnitsAll)) * meanJOverLastTrlFirstSet, sum(nUnitsAll)^2, 1);
%          end
%          
%          if iSet == max(allSetIDs)
%              meanJOverLastTrlLastSet = mean(intraJSample{end}, 3);
%              meanJOverLastTrlLastSetPlot = reshape(sqrt(sum(nUnitsAll)) * meanJOverLastTrlLastSet, sum(nUnitsAll)^2, 1);
%          end
%          
%          save([RNNSampleDir, 'classifier_matrices_IntraOnly', monkey, ssnDate, ...
%              '_set', num2str(iSet), '.mat'], 'activitySample', 'intraJSample', 'trlIDsCurrSet', 'iSet', '-v7.3')
%          
%          save([RNNSampleDir, 'classifier_matrices_resh_IntraOnly', monkey, ssnDate, ...
%              '_set', num2str(iSet), '.mat'], 'activitySample', 'intraJSampleResh', 'trlIDsCurrSet', 'iSet', '-v7.3')
%                   
%          save([RNNSampleDir, 'classifier_matrices_InterOnly', monkey, ssnDate, ...
%              '_set', num2str(iSet), '.mat'], 'activitySample', 'interJSample', 'trlIDsCurrSet', 'iSet', '-v7.3')
%          
%          save([RNNSampleDir, 'classifier_matrices_resh_InterOnly', monkey, ssnDate, ...
%              '_set', num2str(iSet), '.mat'], 'activitySample', 'interJSampleResh', 'trlIDsCurrSet', 'iSet', '-v7.3')
%          
%      end
    
    % plot histograms of Js
    figure,
    [J_bincounts,edgesnew] = histcounts(meanJOverLastTrlFirstSetPlot, 75);
    histcenters = edgesnew(1:end-1) + (diff(edgesnew) ./ 2);
    % normalizing each by area under curve
    semilogy(histcenters,J_bincounts./sum(J_bincounts), 'o-', 'color', [0.12, 0.56, 1], 'linewidth', 1.5, 'markersize', 1.5)
    [J_bincounts,edgesnew] = histcounts(meanJOverLastTrlLastSetPlot, 75);
    histcenters = edgesnew(1:end-1) + (diff(edgesnew) ./ 2);
    hold on, semilogy(histcenters,J_bincounts./sum(J_bincounts), 'o-', 'color', [0 0 1], 'linewidth', 1.5, 'markersize', 1.5)
    set(gca, 'fontsize', 13, 'ylim', [0.0001 1], 'xlim', [-20 20])
    xlabel('Weight')
    ylabel('Density')
    lgd = legend({'first set', 'last set'}, 'location', 'northeast');
    print(gcf, '-dtiff', '-r400', [RNNfigdir, 'intraRgnDeltaJs_consecutiveSamplesSingleTrl', filesep, monkey, ssnDate, '_firstLastHists']),
    close     
           
    meanFRAcrossTrls = mean(cell2mat(arrayfun(@(iTrl) mean(activitySampleAll{iTrl}, 1), 1 : nTrls, 'un', false)'), 1);
    stdFRAcrossTrls = mean(cell2mat(arrayfun(@(iTrl) std(activitySampleAll{iTrl}, [], 1), 1 : nTrls, 'un', false)'), 1); % std over all units, averaged over all trials
    
    %% IMSHOW DELTA J (INTRA ONLY) BETWEEN SAMPLES IN A SINGLE TRIAL
    
    tmpInds = find(trlInfo.trls_since_nov_stim == 0);
    
    while trlInfo.nov_stim_rwd_prob(tmpInds(setToPlot - 1)) == 999
        setToPlot = setToPlot + 1; % move down sets until the last one of prev trial is 999 (get it at most novel part of session
    end
    
    trlToPlot = tmpInds(setToPlot); % first trial of first full set (where the three stimuli on the screen have not yet been learned) - TO DO: is set 1 or set 2 more appropriate?
    
    subT = trlInfo(trlToPlot, :); % to cross reference and make sure you're pulling Js from the trials you want
    assert(sum(subT.trls_since_nov_stim == 0) == 1) % should only be one instance of novel stim appearing

    eventInds = round(subT.aligned_event_times / dtData); % fixation, stim, choice, outcome (relative to fixation)
    sampleEventInds = eventInds(2 : end) - eventInds(2) + 1; % stim, choice, outcome (relative to sample inds for J which start from stim)
    sampleInds = sampleTimePoints - sampleTimePoints(1) + 1; % relative to stim
    
    % find closest match between sampleEventInds and sampleTimePoints
    eventPos = arrayfun(@(iEvent) find(abs(sampleInds - sampleEventInds(iEvent)) == min(abs(sampleInds - sampleEventInds(iEvent)))), 1 : length(sampleEventInds));
    eventLabel = {'stim', 'choice', 'outcome'};
    
    cm = brewermap(100, '*RdBu');
    nDSamples = nSamples - 1;
    if ~isfolder([RNNfigdir, 'intraRgnDeltaJs_consecutiveSamplesSingleTrl', filesep])
        mkdir([RNNfigdir, 'intraRgnDeltaJs_consecutiveSamplesSingleTrl', filesep])
    end
    
    figT = figure('color','w');
    AxT = arrayfun(@(i) subplot(sum(rgnIxToPlot & ~badRgn), nDSamples, i, 'NextPlot', 'add', 'Box', 'on', 'BoxStyle', 'full', 'linewidth', 1, ...
        'xtick', '', 'xticklabel', '', ...
        'ytick', '', 'ydir', 'reverse'), 1:((sum(rgnIxToPlot & ~badRgn) * nDSamples)));
    set(gcf, 'units', 'normalized', 'outerposition', [0 0.1 1 0.9])
    count = 1;
    eventCount = 1;
    dJ_consecutive = [];
    colormap(cm)
    
    for iRgn = find(rgnIxToPlot & ~badRgn) + 1
        if ~ismember(iRgn - 1, find(rgnIxToPlot)) || ismember(iRgn - 1, find(badRgn))
            continue
        end
        
        in_rgn = newRgnInds(iRgn - 1) + 1 : newRgnInds(iRgn);
        nUnitsRgn = numel(in_rgn);
        
        % get difference matrix (changes in intraregional J) for desired trials
        deltaIntraRgnConsJ = diff(cell2mat(arrayfun(@(iTrl) intraJ{iTrl}(in_rgn, in_rgn, :), trlToPlot, 'un', 0)), 1, 3);
        

        % try to sort on descending max mean value based
        % on last delta
        deltaToSort = squeeze(deltaIntraRgnConsJ(:, :, 1));
        
        [~, preSynSort] = sort(mean(deltaToSort, 1), 'descend');
        [~, postSynSort] = sort(mean(deltaToSort, 2), 'descend');
        
        % reshape for PCA
        deltaConsJResh = cell2mat(arrayfun(@(iSample) ...
            reshape(squeeze(deltaIntraRgnConsJ(:, :, iSample)), 1, nUnitsRgn^2), 1 : nDSamples, 'un', false)');
        
        % test for unreshaping
        deltaJConsUnresh = reshape(deltaConsJResh(1, :), [nUnitsRgn, nUnitsRgn]);
        assert(isequal(squeeze(deltaIntraRgnConsJ(:, :, 1)), deltaJConsUnresh))
        
        for i = 1 : nDSamples
            
            J_delta = squeeze(deltaIntraRgnConsJ(:, :, i)); % J_curr - J_prev;
            
            % rearrange to try to find potential structure in terms of
            % changes
            J_delta = J_delta(postSynSort, postSynSort); % makes sense to look at this since that's what gets changed....
            
            dJ_consecutive = [dJ_consecutive; J_delta(:)];
            subplot(AxT(count)), imagesc(J_delta); axis tight
            
            pos = get(AxT(count), 'position');
            pos(3) = 1 ./ (1.25 * nDSamples);% 1.2 * pos(3); % adjust width
            
            if count > 1
                prev_pos = get(AxT(count - 1), 'position');
            end
            
            if mod(count - 1, nDSamples) == 0 || count == 1
                pos(1) = 0.2 * pos(1); % first plot of a row moves to L
            else
                pos(1) = prev_pos(1) + prev_pos(3) + 0.2*pos(3); % adjust L position based on width of plots in earlier row
            end
            
            set(AxT(count), 'position', pos)
            
            if count == 1
                presynLbl = text(AxT(count), round(nUnitsRgn/3), round(-1 * nUnitsRgn/12.5), 'pre-syn', 'fontsize', 10, 'fontweight', 'bold');
                postsynLbl = text(AxT(count), round(1.08*nUnitsRgn), round(nUnitsRgn/3), 'post-syn', 'fontsize', 10, 'fontweight', 'bold');
                set(postsynLbl, 'rotation', 270, 'horizontalalignment', 'left')
            end
            
            if i == 1
                ylabel([rgnLabels{iRgn - 1}, ' (', num2str(nUnitsRgn), ' units)'], 'fontweight', 'bold', 'fontsize', 13)
                set(get(AxT(count), 'ylabel'), 'rotation', 90, 'horizontalalignment', 'center')
            end
            
            if count >= length(AxT) - (nDSamples) + 1
                if ismember(i, eventPos)
                    xlabel({[num2str(i + 1) '\Delta', num2str(i)]; ['(' eventLabel{eventCount} ')']}, 'fontsize', 12, 'fontweight', 'bold')
                    eventCount = eventCount + 1;
                else
                    xlabel([num2str(i + 1), '\Delta', num2str(i)], 'fontsize', 12, 'fontweight', 'bold')
                end
            end
            
            count = count + 1;
            
        end
    end
    
    % update clims for the last three
    [~, L, U] = isoutlier(dJ_consecutive, 'percentile', [0.5 99.5]);
    
    newCLims = [-1 * round(max(abs([L, U])), 5, 'decimals'), round(max(abs([L, U])), 5, 'decimals')];
    set(AxT, 'clim', newCLims),
    titleAxNum = round(0.5*(nDSamples));
    text(AxT(titleAxNum ), -300, -15, ...
        [monkey, ssnDate,': consecutive within-trl \DeltaJs trl=', num2str(trlToPlot), ', ([0.5 99.5] %ile shown (range=\Delta +/-', num2str(newCLims(2), '%.1e'), ') --- SORTED ON: 1st \Delta (postsyn)'], 'fontweight', 'bold', 'fontsize', 14)
    
    set(figT, 'currentaxes', AxT),
    print(figT, '-dtiff', '-r400', [RNNfigdir, 'intraRgnDeltaJs_consecutiveSamplesSingleTrl', filesep, monkey, ssnDate, '_intraRgnDeltaJs_withinFirstTrl']),
    close(figT)
    %% WIP: PCA on consecutive intraregion Js (all samples, all trials)
    
    intraRgnConsJToPlot = intraJ; 
    intraRgnConsJToPlot = cat(3, intraRgnConsJToPlot{:});
    intraJAllResh = cell2mat(arrayfun(@(i) reshape(squeeze(intraRgnConsJToPlot(:, :, i)), 1, sum(nUnitsAll)^2), 1 : (nTrls * nSamples), 'un', false)');
    
    jMean = mean(intraJAllResh, 1); % mean(cell2mat(arrayfun(@(iTrl) mean(cell2mat(intraRgnConsJResh{iTrl}'), 1), 1 : nTrls, 'un', 0)'), 1);
    [~, upperJ_cutoff, lowerJ_cutoff, C] = isoutlier(jMean(jMean~=0), 'percentile', [47.5 52.5]); % remove middle 5% of elements from the mean of all Js (all samples all trials)
    
    % remove middle 5% average of weights
    rmEls = jMean > upperJ_cutoff & jMean < lowerJ_cutoff;
    intraJReshCulled = intraJAllResh;
    intraJReshCulled(:, rmEls) = [];
    
    [w, scores, eigen, ~, pvar, mu] = pca(intraJReshCulled, 'Algorithm', 'svd', 'Centered', center_data, 'Economy', 0);
    projData = intraJReshCulled;
    tempProj = (projData - repmat(mu,size(projData,1),1)) * w;
    x = tempProj;
    
    figure('color', 'w'); hold on
    set(gcf, 'units', 'normalized', 'outerposition', [0.25 0.1 0.5 0.8])
    cm = colormap(cool);
    colormap(cm)
    plot3(x(:, 1), x(:, 2), x(:, 3), 'color', [0.66 0.66 0.66], 'linewidth', 2)
    plot3(x(1 : nSamples : end, 1), x(1 : nSamples : end , 2), x(1 : nSamples : end, 3), 'linestyle', 'none', 'marker', 'o', 'markersize', 8, 'markeredgecolor', [0 0 0.2], 'markerfacecolor', 'w')
    set(gca, 'fontweight', 'bold', 'fontsize', 13)
    xlabel('PC 1'), ylabel('PC 2'), zlabel('PC 3')
    grid minor
    view(3)

    %
            
    figE = figure('color','w');
    set(gcf, 'units', 'normalized', 'outerposition', [0.1 0 0.8 1])
    AxE = arrayfun(@(i) subplot(nRegions/2, 2, i, 'NextPlot', 'add', 'fontweight', 'bold', 'fontsize', 12, ...
        'linewidth', 1.5, 'ylim', [0 110]), 1 : nRegions - sum(badRgn));
    
    figP = figure('color', 'w'); hold on
    set(gcf, 'units', 'normalized', 'outerposition', [0.1 0 0.8 1])
    AxP = arrayfun(@(i) subplot(nRegions/2, 2, i, 'NextPlot', 'add', 'fontweight', 'bold', 'fontsize', 12, ...
        'linewidth', 1.5), 1 : nRegions - sum(badRgn));
    
    count = 1;
    
    subJReshAll = [];
    wRgns = cell(numel(find(~badRgn)), 1);
    pjnRgns = cell(numel(find(~badRgn)), 1);
    
    for iRgn = find(~badRgn)
        
        in_rgn = newRgnInds(iRgn) + 1 : newRgnInds(iRgn + 1);
        
        nUnitsRgn = numel(in_rgn);
        assert(isequal(nUnitsRgn, nUnitsAll(iRgn)))
        
        subJ = intraRgnConsJToPlot(in_rgn, in_rgn, :); % all trials, all samples, intraRgn for a given region
        
        % reshape for PCA
        subJResh = cell2mat(arrayfun(@(iTrl) ...
            reshape(squeeze(subJ(:, :, iTrl)), 1, nUnitsRgn^2), 1 : size(subJ, 3), 'un', false)');
      
        subJReshAll = [subJReshAll, subJResh];
        
        % test for unreshaping
        assert(isequal(squeeze(subJ(:, :, 1)), reshape(subJResh(1, :), [nUnitsRgn, nUnitsRgn])))
        
        % coeffs from all elements of J for each delta
        [w, scores, eigen, ~, pvar, mu] = pca(subJResh, 'Algorithm', 'svd', 'Centered', center_data, 'Economy', 0);
        
        wRgns{iRgn} = w;
        
        % project each regional subspace onto all data
        pjnRgns{iRgn} = (subJResh - repmat(mu,size(subJResh,1),1)) * w;
        
        % estimate effective dimensionality
        set(0, 'currentfigure', figE);
        subplot(AxE(count)), plot(1 : length(eigen), cumsum(pvar), 'linewidth', 1.5, 'color', 'b')
        grid minor, xlabel('# PCs (5% of actual total)'), ylabel('% var explained')
        xlim(AxE(count), [1 0.05 * round(size(subJResh, 2))])
        nPC_cutoff = find(cumsum(pvar) >= 99,1, 'first');
        line(gca, [nPC_cutoff nPC_cutoff], get(gca, 'ylim'), 'linestyle', '--', 'linewidth', 1.5, 'color', 'k')
        text(gca, round(1.2 * nPC_cutoff), 10, ['#PCs = ', num2str(nPC_cutoff)], 'fontweight', 'bold')
        title(AxE(count), rgnLabels{iRgn})
        
        % project  Js back into low D space 
        projData = subJResh;
        tempProj = (projData - repmat(mu,size(projData,1),1)) * w;
        x = tempProj;
        set(0, 'currentfigure', figP);
        subplot(AxP(count)),
        plot3(x(:, 1), x(:, 2), x(:, 3), 'linestyle', rgnLinestyle{iRgn}, 'color', rgnColors(iRgn, :), 'linewidth', 1.75)
        plot3(x(nSamples : nSamples : end - nSamples, 1), x(nSamples : nSamples : end - nSamples, 2), x(nSamples : nSamples : end - nSamples, 3), 'linestyle', 'none', 'marker', 'o', 'markersize', 6, 'markeredgecolor', 'k', 'markerfacecolor', rgnColors(iRgn, :))
        plot3(x(1, 1), x(1, 2), x(1, 3), 'linestyle', 'none', 'marker', 'd', 'markersize', 12, 'markeredgecolor', 'k', 'markerfacecolor', rgnColors(iRgn, :))
        plot3(x(end, 1), x(end, 2), x(end, 3), 'linestyle', 'none', 'marker', 's', 'markersize', 12, 'markeredgecolor', 'k', 'markerfacecolor', rgnColors(iRgn, :))
        title(AxP(count), rgnLabels{iRgn})
        xlabel('PC 1'), ylabel('PC 2'), zlabel('PC 3')
        grid minor
        view(3)
        
        count = count + 1;
    end
    
    % Low D projection
    if ~isfolder([RNNfigdir, 'topPCsIntraRgnJs_consecutive', filesep])
        mkdir([RNNfigdir, 'topPCsIntraRgnJs_consecutive', filesep])
    end
    
    set(0, 'currentfigure', figP);
    xL = max(max(abs(cell2mat(get(AxP, 'xlim')))));
    yL = max(max(abs(cell2mat(get(AxP, 'ylim')))));
    zL = max(max(abs(cell2mat(get(AxP, 'zlim')))));
    set(AxP, 'xlim', [-xL xL], 'ylim', [-yL yL], 'zlim', [-zL zL])
    text(AxP(1), 3.5 * max(get(AxP(1), 'xlim')), 3.5 * max(get(AxP(1), 'ylim')), [monkey, ssnDate, ': consecutive Js (#trials=', num2str(size(subJ, 3)), ') projected into top 3 PCs'], 'fontsize', 14, 'fontweight', 'bold')
    print('-dtiff', '-r400', [RNNfigdir, 'topPCsIntraRgnJs_consecutive', filesep, monkey, ssnDate, '_topPCsIntraRgnJs_consecutive'])
    close
    
    
    % Effective dimensionality
    if ~isfolder([RNNfigdir, 'effDimIntraRgnJs_consecutive', filesep])
        mkdir([RNNfigdir, 'effDimIntraRgnJs_consecutive', filesep])
    end
    
    set(0, 'currentfigure', figE);
    text(AxE(1), 0.75 * max(get(AxE(1), 'xlim')), 130, [monkey, ssnDate, ': eff dim (99% var exp) for consecutive Js (#trials=', num2str(size(subJ, 3)), ')'], 'fontsize', 14, 'fontweight', 'bold')
    print('-dtiff', '-r400', [RNNfigdir, 'effDimIntraRgnJs_consecutive', filesep, monkey, ssnDate, '_effDimIntraRgnJs_consecutive'])
    close
    

    % same as above but all in one axes and without centering
    figure('color','w'); hold on,
    set(gcf, 'units', 'normalized', 'outerposition', [0.2 0.15 0.6 0.7])
    set(gca, 'fontweight', 'bold', 'fontsize', 13)
    
    for iRgn = find(~badRgn)
        PC1 = pjnRgns{iRgn}(:, 1);
        PC2 = pjnRgns{iRgn}(:, 2);
        PC3 = pjnRgns{iRgn}(:, 3);
        
        %PC1 = PC1 - PC1(1);
        %PC2 = PC2 - PC2(1);
        %PC3 = PC3 - PC3(1);
        all_lines(iRgn) = plot3(PC1,PC2,PC3, 'linestyle', rgnLinestyle{iRgn}, 'color', rgnColors(iRgn, :), 'linewidth', 1.75);
        plot3(PC1(1, :), PC2(1, :), PC3(1, :), 'linestyle', 'none', 'marker', 'd', 'markersize', 10, 'markeredgecolor', 'k', 'markerfacecolor', rgnColors(iRgn, :))
        plot3(PC1(end, :), PC2(end, :), PC3(end, :), 'linestyle', 'none', 'marker', 's', 'markersize', 10, 'markeredgecolor', 'k', 'markerfacecolor', rgnColors(iRgn, :))
        
    end
    
    set(gca, 'xlim', [-xL xL], 'ylim', [-yL yL], 'zlim', [-zL zL])
    grid minor
    view(3)
    xlabel('PC 1'), ylabel('PC 2'), zlabel('PC 3')
    title(gca, [monkey, ssnDate, ': consecutive Js (#trials=', num2str(size(subJReshAll, 1)), ') projected into top 3 PCs'], 'fontsize', 14, 'fontweight', 'bold')
    legend(all_lines, rgnLabels(~badRgn), 'location', 'bestoutside')
    print('-dtiff', '-r400', [RNNfigdir, 'topPCsIntraRgnJs_consecutive', filesep, monkey, ssnDate, '_topPCsIntraRgnJs_consecutive_v2'])
    close
%     
    % try PCA on full J for consecutive trials
    %     nUnits = size(intraRgnJ, 1);
    %     consecutiveJ = intraRgnJ(:, :, trlsToPlot);
    %     consecutiveJMean = mean(consecutiveJ, 3);
    %     rmElementsFull = (consecutiveJMean > upperJ_cutoff & consecutiveJMean < lowerJ_cutoff);
    %
    %     consecutiveJResh = cell2mat(arrayfun(@(iTrl) ...
    %         reshape(squeeze(consecutiveJ(:, :, iTrl)), 1, nUnits^2), 1 : size(consecutiveJ, 3), 'un', false)');
    %
    %     % remove a bunch of elements
    %     consecutiveJResh(:, rmElementsFull) = [];
    

    
end

