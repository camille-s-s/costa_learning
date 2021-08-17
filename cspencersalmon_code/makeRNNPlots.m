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
setToPlot       = 2; % only one for now - must be 2 or greater to get last trial prev set

% in and outdirs
bd              = '~/Dropbox (BrAINY Crew)/costa_learning/';
mdlDir          = [bd 'models/'];
RNNfigdir       = [bd 'figures/'];
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

for iFile = 1 : length(allFiles) % for each session....
    
    fName = allFiles(iFile).name;
    fID = fName(1:strfind(fName, '_') - 1);
    monkey = fID(1);
    ssnDate = fID(2:end);
    
    cd([mdlDir, monkey, ssnDate, filesep])
    currSsn = dir('rnn_*_set*_trial*.mat');
    
    % currSsn = dir(['rnn_', allFiles{iFile}, '_*.mat']);
    
    allSetIDs = unique(arrayfun(@(i) ...
        str2double(currSsn(i).name(strfind(currSsn(i).name,'set') + 3 : strfind(currSsn(i).name,'trial') - 2)), 1:length(currSsn)));
    allTrialIDs = unique(arrayfun(@(i) ...
        str2double(currSsn(i).name(strfind(currSsn(i).name,'trial') + 5 : end - 4)), 1:length(currSsn)));
    
    nTrls = length(allTrialIDs);
    nSets = length(allSetIDs);
    
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
        RNN.mdl.params.nRunTot == 1515)
    
    if isfield(RNN.mdl.params, 'smoothWidth')
        assert(RNN.mdl.params.smoothWidth == 0.15)
    end
    
    % load full targets that went into fitCostaRNN for sanity checking with
    % targets from each trial's model
    metafnm = [spikeInfoPath, monkey, ssnDate, '_spikesCont'];
    S = load(metafnm);
    targets = S.dsAllSpikes;
    
    %% recreate target prep from fitCostaRNN for comparison of targets from each trial
    
    if RNN.mdl.params.doSmooth
        targets = smoothdata(targets, 2, 'gaussian', 0.15 / RNN.mdl.dtData); % convert smoothing kernel from msec to #bins);
    end
    
    % this will soft normalize a la Churchland papers
    if RNN.mdl.params.doSoftNorm
        normfac = range(targets, 2); % + (dtData * 10); % normalization factor = firing rate range + alpha
        targets = targets ./ normfac;
    end
    
    if RNN.mdl.params.normByRegion
        arrayList = unique(spikeInfo.array);
        nArray = length(arrayList);
        
        for aa = 1:nArray
            inArray = strcmp(spikeInfo.array,arrayList{aa});
            
            arraySpikes = targets(inArray, :);
            targets(inArray,:) = arraySpikes ./ max(max(arraySpikes));
        end
    else
        targets = targets ./ max(max(targets));
    end
    
    if RNN.mdl.params.rmvOutliers
        
        outliers = isoutlier(mean(targets, 2), 'percentiles', [0.5 99.5]);
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
    
    % housekeeping
    if any(isnan(targets(:)))
        keyboard
    end
    
    %%
    % set up indexing vectors for submatrices
    rgns            = RNN.mdl.params.arrayRegions;
    dtData          = RNN.mdl.dtData;
    arrayList       = rgns(:, 2);
    nRegions        = length(arrayList);
    rgnColors       = brewermap(nRegions, 'Spectral');% cmap(round(linspace(1, 255, nRegions)),:);
    JDim            = size(RNN.mdl.J);
    clear RNN
    
    % collect outputs for all trials in a session
    trlNum              = NaN(1, nTrls);
    setID               = NaN(1, nTrls);
    R_U                 = cell(nTrls, 1);
    R_S                 = cell(nTrls, 1);
    J_U                 = NaN([JDim, nTrls]);
    J_S                 = NaN([JDim, nTrls]);
    J0_U                = NaN([JDim, nTrls]);
    J0_S                = NaN([JDim, nTrls]);
    D_U                 = cell(nTrls, 1);
    D_S                 = cell(nTrls, 1);
    
    tic
    for i = 1 : nTrls
        
        % TO DO: check that all dtRNN are the same
        mdlfnm = dir([mdlDir, monkey, ssnDate, filesep, 'rnn_', monkey, ssnDate, '_set*_trial', num2str(i), '.mat']);
        tmp = load(mdlfnm.name);
        mdl             = tmp.RNN.mdl;
        
        assert(mdl.dtRNN == 0.0005 & ...
            mdl. dtData == 0.01)
        
        iTarget         = mdl.iTarget;
        
        % turn a scrambly boy back into its OG order - target_unit_order(iTarget) = R_and_J_unit_order
        % aka targets and regions are unscrambled, R and J are scrambled
        %         B,I] = sort(A); %
        %         [~,X] = sort(I);
        %         isequal(A, B(X));
        [~, unscramble] = sort(iTarget); % Gives you indices from scrambled to unscrambled
        
        trlNum(i)       = mdl.iTrl;
        setID(i)        = mdl.setID;
        R_S{i}          = mdl.RMdlSample;
        J_S(:, :, i)    = mdl.J;
        J0_S(:, :, i)   = mdl.J0;
        D_S{i}          = mdl.targets(iTarget, :);
        D_U{i}          = mdl.targets;
        assert(isequal(mdl.targets, D_S{i}(unscramble, :))); % indices from scrambled back to original
        
        % collect first and last to assess convergence
        if i == 1
            firstPVars  = mdl.pVars;
            firstChi2   = mdl.chi2;
        end
        
        if i == nTrls
            lastPVars   = mdl.pVars;
            lastChi2    = mdl.chi2;
        end
        
        % TO DO 8/5 TEST THIS use to unscramble these so that their unit order matches indexing
        % vectors from spikeInfo (which match unscrambled targets/D in the
        % row space)
        J_U(:, :, i)    = mdl.J(unscramble, unscramble);
        J0_U(:, :, i)   = mdl.J0(unscramble, unscramble);
        R_U{i}          = mdl.RMdlSample(unscramble, :); % since targets(iTarget) == R
        % figure, scatter(mean(mdl.targets, 2), mean(mdl.RMdlSample(X, :), 2)), xlabel('targets unscambled'), ylabel('R unscrambled')

    end
    toc
    
    % TO DO: ADD COMPARISON OF JTENSOR WITH EXTRACTED J, ALSO ASSERT THAT
    % ALL J0s ARE DIFFERENT
    % isequal(J0_U(:, :, 2), J_U(:, :, 1))
    JTensor = [mdlDir, monkey, ssnDate, filesep, 'rnn_', monkey, ssnDate, '_JTrl.mat'];
    if exist(JTensor, 'file')
        load(JTensor)
        assert(isequal(J_S, JAllTrl(:, :, 1 : nTrls)))
    end
    
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
    
    setID = setID(trlSort);
    D_S = D_S(trlSort); %  d = cell2mat(D_U'); isequal(d(iTarget, :), cell2mat(D_S'))
    D_U = D_U(trlSort);
    J_S = J_S(:, :, trlSort); % j = squeeze(mean(J_U, 3)), isequal(j(iTarget, iTarget), squeeze(mean(J_S, 3)))
    J0_S = J0_S(:, :, trlSort);
    R_S = R_S(trlSort); % r = cell2mat(R_U'); isequal(r(iTarget, :), cell2mat(R_S'))
    
    % NOTE: USE unscrambled J/J0/R IF YOU WANNA USE SPIKEINFO TO INDEX!
    J_U = J_U(:, :, trlSort);
    J0_U = J0_U(:, :, trlSort);
    R_U = R_U(trlSort);
    
    % compute some useful quantities. note that _U is in same unit order as
    % spikeInfo
    trlDursRNN = arrayfun(@(iTrl) size(D_U{iTrl}, 2), 1 : nTrls)';
    [~, ia, ic] = unique(setID(setID ~= 0)');
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
        disp('might have incomplete set or not enough trials...')
        keyboard
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
    
    %% reorder J by region order (for J plotting of full matrix)
    
    % self proof that this works
    % n = 4;
    % iTarget = randperm(n);
    % A = magic(n);
    % Ap = A(iTarget, iTarget);
    % isequal(A, Ap(iTarget, iTarget))
    
    intraRgnJ = zeros(size(J_U));
    count = 1;
    nUnitsAll = NaN(nRegions, 1);
    newOrder = [];
    
    for iRgn = 1 : nRegions
        in_rgn = rgns{iRgn,3}; % TO DO NEXT: COMPARE D_sort intraregional variance indexed w ogRgns vs newRgns (in terms of variance of D_sort
        newOrder = [newOrder; find(in_rgn)]; % reorder J so that rgns occur in order
        nUnitsRgn = sum(in_rgn);
        nUnitsAll(iRgn) = nUnitsRgn;
        newIdx = count : count + nUnitsRgn - 1;
        intraRgnJ(newIdx, newIdx, :) = J_U(in_rgn, in_rgn, :); % intra-region only
        count = count + nUnitsRgn;
    end
    
    fullRgnJ = J_U(newOrder, newOrder, :); % includes interactions
    
    
    %% pVar / chi2 for first and last J
    
    if ~isfolder([RNNfigdir, 'convergence', filesep])
        mkdir([RNNfigdir, 'convergence', filesep])
    end
    idx = randi(size(intraRgnJ, 1));
    figure('color', 'w')
    set(gcf, 'units', 'normalized', 'outerposition', [0.1 0.15 0.8 0.7])
    nRun = size(firstPVars, 2);
    subplot(2,4,1);
    hold on;
    imagesc(D_U{1}); colormap(jet), colorbar;
    axis tight; set(gca, 'clim', [0 1])
    title('real');
    set(gca,'Box','off','TickDir','out','FontSize',14);
    subplot(2,4,2);
    hold on;
    imagesc(R_U{1}); colormap(jet), colorbar;
    axis tight; set(gca, 'clim', [0 1])
    title('model');
    set(gca,'Box','off','TickDir','out','FontSize',14);
    subplot(2,4,[3 4 7 8]);
    hold all;
    plot(1 : size(R_U{1}, 2), R_U{1}(idx, :), 'linewidth', 1.5);
    plot(1 : size(D_U{1}, 2), D_U{1}(idx, :), 'linewidth', 1.5);
    axis tight; set(gca, 'ylim', [-0.1 1])
    ylabel('activity');
    xlabel('time (s)'),
    legend('model','real','location','eastoutside')
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
    legend('model','real','location','eastoutside')
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
    
    %% example graph (WIP)
    %     nodeNames = repelem(rgnLabels, nUnitsAll);
    %     Jt = mean(Jtmp2, 3);
    %     [~, L, U, C] = isoutlier(Jt(:), 'percentile', [47.5 52.5]);
    %     Jt(Jt > L & Jt < U) = 0; % idk remove middle five percent
    %     figure,
    %     figG = plot(digraph(Jt));
    %
    %     cm = colormap(jet);
    %     cm = cm(round(linspace(1, size(cm, 1), nRegions)), :);
    %
    %     for iRgn = 1 : nRegions
    %         highlight(figG, find(strcmp(nodeNames, rgnLabels{iRgn})), 'markersize', 10, 'nodecolor', cm(iRgn, :));
    %     end
    
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
    
    % elapsed time between events (1 = fixation, 2 = stim, 3 = choice, 4 = outcome, 5 = time of next trl
    % fixation)
    avgEventTS = [mean(T.aligned_event_times, 1), nanmean(durBtwnTrls)];
    eventInds = round(avgEventTS / dtData);
    eventInds(1) = 1;
    
    % pad remaining units and trials to same length
    longestTrl = max(arrayfun(@(s) max(size(Dtmp{s}, 2)), 1 : size(Dtmp, 1)));
    trlsPadded = arrayfun(@(s) ...
        [Dtmp{s}, NaN(size(Dtmp{s}, 1), longestTrl - size(Dtmp{s}, 2))], 1 : size(Dtmp, 1), 'un', false);
    X = cell2mat(trlsPadded);
    
    % trial average them
    trlsAvg = cell2mat(arrayfun( @(n) nanmean(reshape(X(n, :), longestTrl, (nTrls - sum(bad_trls))), 2), 1 : size(X, 1), 'un', false))';
    trlsAvg(bad_units, :) = [];
    
    % coeffs from all activity, trial averaged and padded
    [w, scores, eigen, ~, pvar, mu] = pca(trlsAvg', 'Algorithm', 'svd', 'Centered', center_data, 'Economy', 0);
    
    % estimate effective dimensionality
    
    if ~isfolder([RNNfigdir, 'effDimAllActivity', filesep])
        mkdir([RNNfigdir, 'effDimAllActivity', filesep])
    end
    
    figure('color', 'w'), plot(1 : length(eigen), cumsum(pvar), 'linewidth', 1.5, 'color', 'b')
    set(gca, 'fontweight', 'bold', 'fontsize', 13, 'ylim', [0 110]),
    grid minor, xlabel('# PCs'), ylabel('% variance explained')
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
    projData = trlsAvg';
    tempProj = (projData - repmat(mu,size(projData,1),1)) * w;
    x = tempProj;
    plot3(x(:, 1), x(:, 2), x(:, 3), 'color', 'k', 'linewidth', 1.5)
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
    title(gca, [monkey, ssnDate, ': avg activity from first # ', num2str(nTrls - sum(bad_trls)), ' trls projected into top 3 PCs'], 'fontsize', 14, 'fontweight', 'bold')
    print('-dtiff', '-r400', [RNNfigdir, 'topPCsAllActivity', filesep, monkey, ssnDate, '_topPCsAllActivity'])
    close
    
    %% set up for future J plots reordered so within-rgn is on-diagonal and between-rgn is off-diagonal
    
    tmp = [0; nUnitsAll];
    JLblPos = arrayfun(@(iRgn) sum(tmp(1 : iRgn-1)) + tmp(iRgn)/2, 2 : nRegions); % for the labels separating submatrices
    JLinePos = cumsum(nUnitsAll(~badRgn))'; % for the lines separating regions
    newRgnInds = [0, JLinePos];
    rgnLabels = rgnLabels(~badRgn); % remove bad region
    nUnitsAll = nUnitsAll(~badRgn);
    rgnIxToPlot = rgnIxToPlot(~badRgn);
    
    %% plot delta for intraregional J for consecutive trials from set
    
    % for plotting consecutive trials (last trial prev set is first - to
    % compare delta between sets) as well as delta within set
    tmpInds = find(trlInfo.trls_since_nov_stim == 0);
    
    while trlInfo.nov_stim_rwd_prob(tmpInds(setToPlot - 1)) == 999
        setToPlot = setToPlot + 1; % move down sets until the last one of prev trial is 999 (get it at most novel part of session
    end
    
    trlsToPlot = tmpInds(setToPlot - 1) - 1 : tmpInds(setToPlot - 1) - 1 + nD;
    
    subT = trlInfo(trlsToPlot, :); % to cross reference and make sure you're pulling Js from the trials you want
    assert(sum(subT.trls_since_nov_stim == 0) == 1) % should only be one instance of novel stim appearing
    cm = brewermap(100, '*RdBu');
    
    if ~isfolder([RNNfigdir, 'intraRgnDeltaJs_consecutive', filesep])
        mkdir([RNNfigdir, 'intraRgnDeltaJs_consecutive', filesep])
    end
    
    figT = figure('color','w');
    AxT = arrayfun(@(i) subplot(sum(rgnIxToPlot), nD, i, 'NextPlot', 'add', 'Box', 'on', 'BoxStyle', 'full', 'linewidth', 1, ...
        'xtick', '', 'xticklabel', '', ...
        'ytick', '', 'ydir', 'reverse'), 1:((sum(rgnIxToPlot) * nD)));
    set(gcf, 'units', 'normalized', 'outerposition', [0 0.1 1 0.9])
    count = 1;
    dJ_consecutive = [];
    colormap(cm)
    
    for iRgn = find(rgnIxToPlot) + 1
        if ~ismember(iRgn - 1, find(rgnIxToPlot))
            continue
        end
        
        in_rgn = newRgnInds(iRgn - 1) + 1 : newRgnInds(iRgn);
        nUnitsRgn = numel(in_rgn);
        assert(isequal(nUnitsRgn, nUnitsAll(iRgn - 1)))
        
        % get difference matrix (changes in intraregional J) for desired trials
        dJ = diff(intraRgnJ(in_rgn, in_rgn, trlsToPlot), 1, 3);
        
        % try to sort on descending max mean value (presyn) based
        % on last delta
        deltaToSort = squeeze(dJ(:, :, 1)); % maybe more interesting since this is where the nov stim appears
        
        [~, preSynSort] = sort(mean(deltaToSort, 1), 'descend');
        [~, postSynSort] = sort(mean(deltaToSort, 2), 'descend');
        
        % reshape for PCA
        dJResh = cell2mat(arrayfun(@(iTrl) ...
            reshape(squeeze(dJ(:, :, iTrl)), 1, nUnitsRgn^2), 1 : nD, 'un', false)');
        
        % test for unreshaping
        dJUnresh = reshape(dJResh(1, :), [nUnitsRgn, nUnitsRgn]);
        assert(isequal(squeeze(dJ(:, :, 1)), dJUnresh))
        
        for i = 1 : nD
            
            J_delta = squeeze(dJ(:, :, i)); % J_curr - J_prev;
            
            % rearrange to try to find potential structure in terms of
            % changes
            J_delta = J_delta(postSynSort, postSynSort); % makes sense to look at this since that's what gets changed....
            
            dJ_consecutive = [dJ_consecutive; J_delta(:)];
            subplot(AxT(count)), imagesc(J_delta); axis tight
            
            pos = get(AxT(count), 'position');
            pos(3) = 1 ./ (1.25 * nD);% 1.2 * pos(3); % adjust width
            
            if count > 1
                prev_pos = get(AxT(count - 1), 'position');
            end
            
            if mod(count - 1, nD) == 0 || count == 1
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
                ylabel([rgnLabels{iRgn-1}, ' (', num2str(nUnitsRgn), ' units)'], 'fontweight', 'bold', 'fontsize', 13)
                set(get(AxT(count), 'ylabel'), 'rotation', 90, 'horizontalalignment', 'center')
            end
            
            if count >= length(AxT) - (nD) + 1
                xlabel([num2str(trlsToPlot(i) + 1), '\Delta', num2str(trlsToPlot(i))], 'fontsize', 12, 'fontweight', 'bold')
                if i == 1
                    text(AxT(count), round(nUnitsRgn/5), round(1.25 * nUnitsRgn), '(\Delta from prev set)', 'fontsize', 10, 'fontweight', 'bold')
                end
            end
            
            count = count + 1;
            
        end
        
    end
    
    % update clims for the last three
    [~, L, U] = isoutlier(dJ_consecutive, 'percentile', [0.5 99.5]);
    % newCLims =  [-0.45 0.45];
    
    newCLims = [-1 * round(max(abs([L, U])), 3, 'decimals'), round(max(abs([L, U])), 3, 'decimals')];
    set(AxT, 'clim', newCLims),
    titleAxNum = round(0.5*(nD));
    text(AxT(titleAxNum ), -95, -15, ...
        [monkey, ssnDate,': consecutive within-set \DeltaJs --- ([0.5 99.5] %ile shown (range=\Delta +/-', num2str(newCLims(2), '%.2f'), ') --- SORTED ON: 1st \Delta (postsyn)'], 'fontweight', 'bold', 'fontsize', 14)
    
    set(figT, 'currentaxes', AxT),
    print(figT, '-dtiff', '-r400', [RNNfigdir, 'intraRgnDeltaJs_consecutive', filesep, monkey, ssnDate, '_intraRgnDeltaJs_consecutive']),
    close(figT)
    
    %% PCA on delta for consecutive trials, intraregionally
    
    tmpInds = find(trlInfo.trls_since_nov_stim == 0);
    while trlInfo.nov_stim_rwd_prob(tmpInds(setToPlot - 1)) == 999
        setToPlot = setToPlot + 1; % move down sets until the last one of prev trial is 999 (get it at most novel part of session
    end
    
    trlsToPlot = tmpInds(setToPlot - 1) - 1 : nTrls; % for this one do as many as you have!
    tmpJ = intraRgnJ(:, :, trlsToPlot);
    
    % using all included Js for a given intraregional matrix, remove the
    % middle five percent (threshold it)
    [~, upperJ_cutoff, lowerJ_cutoff, C] = isoutlier(tmpJ(tmpJ~=0), 'percentile', [47.5 52.5]);
    
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
        
        % get difference matrix (changes in intraregional J) for desired trials
        dJ = diff(intraRgnJ(in_rgn, in_rgn, trlsToPlot), 1, 3);
        subJ = intraRgnJ(in_rgn, in_rgn, trlsToPlot);
        
        % reshape for PCA
        dJResh = cell2mat(arrayfun(@(iTrl) ...
            reshape(squeeze(dJ(:, :, iTrl)), 1, nUnitsRgn^2), 1 : size(dJ, 3), 'un', false)');
        subJResh = cell2mat(arrayfun(@(iTrl) ...
            reshape(squeeze(subJ(:, :, iTrl)), 1, nUnitsRgn^2), 1 : size(subJ, 3), 'un', false)');
        
        subJReshAll = [subJReshAll, subJResh];
        % test for unreshaping
        dJUnresh = reshape(dJResh(1, :), [nUnitsRgn, nUnitsRgn]);
        assert(isequal(squeeze(dJ(:, :, 1)), dJUnresh))
        assert(isequal(squeeze(subJ(:, :, 1)), reshape(subJResh(1, :), [nUnitsRgn, nUnitsRgn])))
        
        % threshold - place NaNs for J vals within cutoff
        % subJResh(:, rmElements) = NaN;
        
        % coeffs from all elements of J for each delta
        % [w, scores, eigen, ~, pvar, mu] = pca(dJResh, 'Algorithm', 'svd', 'Centered', center_data, 'Economy', 0);
        [w, scores, eigen, ~, pvar, mu] = pca(subJResh, 'Algorithm', 'svd', 'Centered', center_data, 'Economy', 0);
        
        wRgns{iRgn} = w;
        
        % project each regional subspace onto all data
        % pjnRgns{iRgn} = (subJResh - repmat(mu,size(subJResh,1),1)) * w;
        pjnRgns{iRgn} = subJResh * w;
        
        % estimate effective dimensionality
        set(0, 'currentfigure', figE);
        subplot(AxE(count)), plot(1 : length(eigen), cumsum(pvar), 'linewidth', 1.5, 'color', 'b')
        grid minor, xlabel('# PCs (5% of actual total)'), ylabel('% var explained')
        xlim(AxE(count), [0 0.05 * round(size(subJResh, 2))])
        nPC_cutoff = find(cumsum(pvar) >= 99,1, 'first');
        line(gca, [nPC_cutoff nPC_cutoff], get(gca, 'ylim'), 'linestyle', '--', 'linewidth', 1.5, 'color', 'k')
        text(gca, round(1.2 * nPC_cutoff), 10, ['#PCs = ', num2str(nPC_cutoff)], 'fontweight', 'bold')
        title(AxE(count), rgnLabels{iRgn})
        
        % project delta Js back into low D space (is this valid since
        % like...it's not time?)
        % projData = dJResh;
        projData = subJResh;
        tempProj = (projData - repmat(mu,size(projData,1),1)) * w;
        x = tempProj;
        set(0, 'currentfigure', figP);
        subplot(AxP(count)),
        plot3(x(:, 1), x(:, 2), x(:, 3), 'color', rgnColors(iRgn, :), 'linewidth', 1.5)
        plot3(x(10 : 10 : end - 10, 1), x(10 : 10 : end - 10, 2), x(10 : 10 : end - 10, 3), 'linestyle', 'none', 'marker', 'o', 'markersize', 7, 'markeredgecolor', 'k', 'markerfacecolor', 'k')
        plot3(x(1, 1), x(1, 2), x(1, 3), 'linestyle', 'none', 'marker', 'd', 'markersize', 10, 'markeredgecolor', 'k', 'markerfacecolor', 'g')
        plot3(x(end, 1), x(end, 2), x(end, 3), 'linestyle', 'none', 'marker', 's', 'markersize', 10, 'markeredgecolor', 'k', 'markerfacecolor', 'r')
        title(AxP(count), rgnLabels{iRgn})
        xlabel('PC 1'), ylabel('PC 2'), zlabel('PC 3')
        grid minor
        view(3)
        
        count = count + 1;
    end
    
    % Effective dimensionality
    if ~isfolder([RNNfigdir, 'effDimIntraRgnJs_consecutive', filesep])
        mkdir([RNNfigdir, 'effDimIntraRgnJs_consecutive', filesep])
    end
    
    xL = max(max(abs(cell2mat(get(AxP, 'xlim')))));
    yL = max(max(abs(cell2mat(get(AxP, 'ylim')))));
    zL = max(max(abs(cell2mat(get(AxP, 'zlim')))));
    
    set(AxP, 'xlim', [-xL xL], 'ylim', [-yL yL], 'zlim', [-zL zL])
    
    set(0, 'currentfigure', figE);
    text(AxE(1), 0.75 * max(get(AxE(1), 'xlim')), 130, [monkey, ssnDate, ': eff dim (99% var exp) for consecutive Js (#trials=', num2str(size(subJ, 3)), ')'], 'fontsize', 14, 'fontweight', 'bold')
    
    % print('-dtiff', '-r400', [RNNfigdir, allFiles{f}, '_effDimIntraRgnDeltaJs_consecutive'])
    print('-dtiff', '-r400', [RNNfigdir, 'effDimIntraRgnJs_consecutive', filesep, monkey, ssnDate, '_effDimIntraRgnJs_consecutive'])
    
    % Low D projection
    if ~isfolder([RNNfigdir, 'topPCsIntraRgnJs_consecutive', filesep])
        mkdir([RNNfigdir, 'topPCsIntraRgnJs_consecutive', filesep])
    end
    
    set(0, 'currentfigure', figP);
    text(AxP(1), 3.5 * max(get(AxP(1), 'xlim')), 3.5 * max(get(AxP(1), 'ylim')), [monkey, ssnDate, ': consecutive Js (#trials=', num2str(size(subJ, 3)), ') projected into top 3 PCs'], 'fontsize', 14, 'fontweight', 'bold')
    
    % print('-dtiff', '-r400', [RNNfigdir, allFiles{f}, '_topPCsIntraRgnDeltaJs_consecutive'])
    print('-dtiff', '-r400', [RNNfigdir, 'topPCsIntraRgnJs_consecutive', filesep, monkey, ssnDate, '_topPCsIntraRgnJs_consecutive'])
    
    close all
    
    % same as above but all in one axes and without centering
    % plot all PCs from all pic subspace
    figure('color','w'); hold on,
    set(gcf, 'units', 'normalized', 'outerposition', [0.2 0.15 0.6 0.7])
    set(gca, 'fontweight', 'bold', 'fontsize', 13)
    
    for iRgn = find(~badRgn)
        PC1 = pjnRgns{iRgn}(:, 1);
        PC2 = pjnRgns{iRgn}(:, 2);
        PC3 = pjnRgns{iRgn}(:, 3);
        all_lines(iRgn) = plot3(PC1,PC2,PC3, 'color', rgnColors(iRgn, :), 'linewidth', 1.5);
        plot3(PC1(1, :), PC2(1, :), PC3(1, :), 'linestyle', 'none', 'marker', 'd', 'markersize', 10, 'markeredgecolor', 'k', 'markerfacecolor', 'g')
        plot3(PC1(end, :), PC2(end, :), PC3(end, :), 'linestyle', 'none', 'marker', 's', 'markersize', 10, 'markeredgecolor', 'k', 'markerfacecolor', 'r')
        
    end
    set(gca, 'xlim', [-xL xL], 'ylim', [-yL yL], 'zlim', [-zL zL])
    grid minor
    view(3)
    xlabel('PC 1'), ylabel('PC 2'), zlabel('PC 3')
    title(gca, [monkey, ssnDate, ': consecutive Js (#trials=', num2str(size(subJReshAll, 1)), ') projected into top 3 PCs'], 'fontsize', 14, 'fontweight', 'bold')
    legend(all_lines, rgnLabels, 'location', 'bestoutside')
    print('-dtiff', '-r400', [RNNfigdir, 'topPCsIntraRgnJs_consecutive', filesep, monkey, ssnDate, '_topPCsIntraRgnJs_consecutive_v2'])
    close
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
    
    
    %%
    [w, scores, eigen, ~, pvar, mu] = pca(subJReshAll, 'Algorithm', 'svd', 'Centered', center_data, 'Economy', 0);
    
    if ~isfolder([RNNfigdir, 'effDimFullIntraRgnJs_consecutive/'])
        mkdir([RNNfigdir, 'effDimFullIntraRgnJs_consecutive/'])
    end
    figure('color','w'); hold on,
    set(gcf, 'units', 'normalized', 'outerposition', [0.2 0.15 0.6 0.7])
    set(gca, 'fontweight', 'bold', 'fontsize', 13)
    plot(1 : length(eigen), cumsum(pvar), 'linewidth', 1.5, 'color', 'b')
    grid minor, xlabel('# PCs (1% of actual total)'), ylabel('% var explained')
    xlim(gca, [0 0.01 * round(size(subJReshAll, 2))]), ylim(gca, [0 110])
    nPC_cutoff = find(cumsum(pvar) >= 99,1, 'first');
    line(gca, [nPC_cutoff nPC_cutoff], get(gca, 'ylim'), 'linestyle', '--', 'linewidth', 1.5, 'color', 'k')
    text(gca, round(1.2 * nPC_cutoff), 10, ['#PCs = ', num2str(nPC_cutoff)], 'fontweight', 'bold')
    title(gca, [monkey, ssnDate, ': consecutive Js (#trials=', num2str(size(subJReshAll, 1)), ') projected into top 3 PCs'], 'fontsize', 14, 'fontweight', 'bold')
    title(gca, [monkey, ssnDate, ': eff dim (99% var exp) for consecutive Js (#trials=', num2str(size(subJReshAll, 1)), ')'], 'fontsize', 14, 'fontweight', 'bold')
    
    print('-dtiff', '-r400', [RNNfigdir, 'effDimFullIntraRgnJs_consecutive', filesep, monkey, ssnDate, '_effDimFullIntraRgnJs_consecutive'])
    close
    
    
    if ~isfolder([RNNfigdir, 'topPCsFullIntraRgnJs_consecutive/'])
        mkdir([RNNfigdir, 'topPCsFullIntraRgnJs_consecutive/'])
    end
    figure('color','w'); hold on,
    set(gcf, 'units', 'normalized', 'outerposition', [0.2 0.15 0.6 0.7])
    set(gca, 'fontweight', 'bold', 'fontsize', 13),
    
    nEl = [0; cumsum(nUnitsAll .^ 2)];
    for iRgn = 2 : size(nEl)
        startRgn = nEl(iRgn - 1) + 1;
        stopRgn = nEl(iRgn) ;
        
        projData = subJReshAll(:, startRgn : stopRgn);
        % tempProj = (projData - repmat(mu(startRgn : stopRgn), size(projData,1),1)) * w(startRgn : stopRgn, startRgn : stopRgn);
        tempProj = (projData - repmat(mu, size(projData,1),1)) * w(startRgn : stopRgn, startRgn : stopRgn);
        
        x = tempProj;
        all_lines(iRgn - 1) = plot3(x(:, 1), x(:, 2), x(:, 3), 'color', rgnColors(iRgn-1, :), 'linewidth', 1.5);
        plot3(x(10 : 10 : end - 10, 1), x(10 : 10 : end - 10, 2), x(10 : 10 : end - 10, 3), 'linestyle', 'none', 'marker', 'o', 'markersize', 7, 'markeredgecolor', 'k', 'markerfacecolor', 'k')
        plot3(x(1, 1), x(1, 2), x(1, 3), 'linestyle', 'none', 'marker', 'd', 'markersize', 10, 'markeredgecolor', 'k', 'markerfacecolor', 'g')
        plot3(x(end, 1), x(end, 2), x(end, 3), 'linestyle', 'none', 'marker', 's', 'markersize', 10, 'markeredgecolor', 'k', 'markerfacecolor', 'r')
        
    end
    set(gca, 'xlim', [-xL xL], 'ylim', [-yL yL], 'zlim', [-zL zL])
    grid minor
    
    view(3)
    xlabel('PC 1'), ylabel('PC 2'), zlabel('PC 3')
    
    title(gca, [monkey, ssnDate, ': consecutive Js (#trials=', num2str(size(subJReshAll, 1)), ') projected into top 3 PCs'], 'fontsize', 14, 'fontweight', 'bold')
    legend(all_lines, rgnLabels, 'location', 'bestoutside')
    print('-dtiff', '-r400', [RNNfigdir, 'topPCsFullIntraRgnJs_consecutive', filesep, monkey, ssnDate, '_topPCsFullIntraRgnJs_consecutive'])
    close
    
    
end

