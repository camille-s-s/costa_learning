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
RNNSampleDir    = [mdlDir filesep 'model_samples/'];
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
        RNN.mdl.params.nRunTot == 1010)
    
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
    rgnColors       = [1 0 0; 1 0 0; 1 0 1; 1 0 1; 0 0 1; 0 0 1; 0 1 0; 0 1 0]; % r, m, b, g brewermap(nRegions, 'Spectral');% cmap(round(linspace(1, 255, nRegions)),:);
    rgnLinestyle    = {'-', '-.', '-', '-.', '-', '-.', '-', '-.'};
    JDim            = size(RNN.mdl.J);
    clear RNN
    
    % collect outputs for all trials in a session
    trlNum              = NaN(1, nTrls);
    prevTrlNum          = NaN(1, nTrls);
    setID               = NaN(1, nTrls);
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
    fittedConsJ         = cell(nTrls, 1);
    
    tic
    
    activitySampleAll      = cell(nTrls, 1);
    consecutiveJSampleAll  = cell(nTrls, 1);
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
        
        trlNum(i)       = mdl.iTrl;
        prevTrlNum(i)   = mdl.prevTrl;
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
     
        pVarsTrls(i)    = mdl.pVars(end);
        chi2Trls(i)     = mdl.chi2(end);
        
        % in the event of permuted iTarget
        J_U(:, :, i)    = mdl.J(unscramble, unscramble);
        J0_U(:, :, i)   = mdl.J0(unscramble, unscramble);
        R_U{i}          = mdl.RMdlSample(unscramble, :); % since targets(iTarget) == R
        % figure, scatter(mean(mdl.targets, 2), mean(mdl.RMdlSample(X, :), 2)), xlabel('targets unscambled'), ylabel('R unscrambled')
        
        % choose ~21 samples per trial from each consecutive J after model
        % convergence (for Nayebi classifier parity)
        fittedConsJ{i}  = mdl.fittedConsJ(:, :, :);
        allSPTimePoints(i, :) = mdl.sampleTimePoints;

    end
    toc
    
    if any(pVarsTrls < 0)
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
    setID = setID(trlSort);
    
    if any(pVarsTrls < 0)
        lastGoodTrl = find(pVarsTrls<0, 1) - 1;
        trlSort = trlSort(1 : lastGoodTrl);
        setID = setID(1 : lastGoodTrl);
        
        % truncate a bit more to last complete set
        trlNumLastGoodSet = find(setID == max(setID) - 1, 1, 'last');
        trlSort = trlSort(1 : trlNumLastGoodSet);
        nTrls = trlNumLastGoodSet;
    end
    
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
    
    % sanity check that all J0s are different
    J0_resh = cell2mat(arrayfun(@(iTrl) ...
        reshape(squeeze(J0_U(:, :, iTrl)), 1, size(J0_U, 1)^2), 1 : size(J0_U, 3), 'un', false)');
    assert(isequal(size(unique(J0_resh, 'rows'), 1), nTrls))
    
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
    tmpintraRgnConsJ = zeros(size(fittedConsJ{1})); % subsampled consecutive fitted Js from second trial (aka first trial of first set)
    intraRgnConsJ = repmat({tmpintraRgnConsJ}, length(fittedConsJ), 1);
    intraRgnConsJResh = cell(nTrls, 1);
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
        
        for iTrl = 1 : length(fittedConsJ)
            
            % re-order by region
            intraRgnConsJ{iTrl}(newIdx, newIdx, :) = fittedConsJ{iTrl}(in_rgn, in_rgn, :); % for subsampled consecutive Js for Nayeli classifier
        
            % vectorize
            
            intraRgnConsJResh{iTrl} = arrayfun(@(iSample) ...
                reshape(squeeze(intraRgnConsJ{iTrl}(:, :, iSample)), 1, size(in_rgn, 1)^2), 1 : nSamples, 'un', false);
        
        end
        
        count = count + nUnitsRgn;
    end
    
    fullRgnJ = J_U(newOrder, newOrder, :); % includes interactions
    fullRgnConsJ = arrayfun(@(iTrl) fittedConsJ{iTrl}(newOrder, newOrder, :), 1 : length(fittedConsJ), 'un', 0)';
    
    
    %% pVar / chi2 for first and last J
    
    if ~isfolder([RNNfigdir, 'pVar', filesep])
        mkdir([RNNfigdir, 'pVar', filesep])
    end
    
    figure, plot(pVarsTrls(trlSort), 'linewidth', 1.5), set(gca, 'ylim', [.90 1], 'xlim', [1 nTrls]),
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
    % fig2svg([RNNfigdir, 'convergence', filesep, monkey, ssnDate, '_firstTrlConvergence.svg'], ...
        % gcf, 1, lgd, [], [], 'jpg')
    close
    
    % last
    if ~any(pVarsTrls < 0)
        
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
        fig2svg([RNNfigdir, 'convergence', filesep, monkey, ssnDate, '_lastTrlConvergence.svg'], ...
            gcf, 1, lgd)
        close
    end
    
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
    
   
    %% set up for future J plots reordered so within-rgn is on-diagonal and between-rgn is off-diagonal
    
    tmp = [0; nUnitsAll];
    JLblPos = arrayfun(@(iRgn) sum(tmp(1 : iRgn-1)) + tmp(iRgn)/2, 2 : nRegions); % for the labels separating submatrices
    JLinePos = cumsum(nUnitsAll)'; % for the lines separating regions
    newRgnInds = [0, JLinePos];
    % rgnLabels = rgnLabels(~badRgn); % remove bad region
    % nUnitsAll = nUnitsAll(~badRgn);
    % rgnIxToPlot = rgnIxToPlot(~badRgn);
    
    %% plot delta for intraregional J for consecutive trials from set
    
    % for plotting consecutive trials (last trial prev set is first - to
     % compare delta between sets) as well as delta within set
%     tmpInds = find(trlInfo.trls_since_nov_stim == 0);
%     
%     while trlInfo.nov_stim_rwd_prob(tmpInds(setToPlot - 1)) == 999
%         setToPlot = setToPlot + 1; % move down sets until the last one of prev trial is 999 (get it at most novel part of session
%     end
%     
%     trlsToPlot = tmpInds(setToPlot - 1) - 1 : tmpInds(setToPlot - 1) - 1 + nD;
%     
%     subT = trlInfo(trlsToPlot, :); % to cross reference and make sure you're pulling Js from the trials you want
%     assert(sum(subT.trls_since_nov_stim == 0) == 1) % should only be one instance of novel stim appearing
%     cm = brewermap(100, '*RdBu');
%     
%     if ~isfolder([RNNfigdir, 'intraRgnDeltaJs_consecutive', filesep])
%         mkdir([RNNfigdir, 'intraRgnDeltaJs_consecutive', filesep])
%     end
%     
%     figT = figure('color','w');
%     AxT = arrayfun(@(i) subplot(sum(rgnIxToPlot & ~badRgn), nD, i, 'NextPlot', 'add', 'Box', 'on', 'BoxStyle', 'full', 'linewidth', 1, ...
%         'xtick', '', 'xticklabel', '', ...
%         'ytick', '', 'ydir', 'reverse'), 1:((sum(rgnIxToPlot & ~badRgn) * nD)));
%     set(gcf, 'units', 'normalized', 'outerposition', [0 0.1 1 0.9])
%     count = 1;
%     dJ_consecutive = [];
%     colormap(cm)
%     
%     for iRgn = find(rgnIxToPlot & ~badRgn) + 1
%         if ~ismember(iRgn - 1, find(rgnIxToPlot)) || ismember(iRgn - 1, find(badRgn))
%             continue
%         end
%         
%         in_rgn = newRgnInds(iRgn - 1) + 1 : newRgnInds(iRgn);
%         nUnitsRgn = numel(in_rgn);
%         % assert(isequal(nUnitsRgn, nUnitsAll(iRgn - 1)))
%         
%         % get difference matrix (changes in intraregional J) for desired trials
%         dJ = diff(intraRgnJ(in_rgn, in_rgn, trlsToPlot), 1, 3);
%         
%         % try to sort on descending max mean value (presyn) based
%         % on last delta
%         deltaToSort = squeeze(dJ(:, :, 1)); % maybe more interesting since this is where the nov stim appears
%         
%         [~, preSynSort] = sort(mean(deltaToSort, 1), 'descend');
%         [~, postSynSort] = sort(mean(deltaToSort, 2), 'descend');
%         
%         % reshape for PCA
%         dJResh = cell2mat(arrayfun(@(iTrl) ...
%             reshape(squeeze(dJ(:, :, iTrl)), 1, nUnitsRgn^2), 1 : nD, 'un', false)');
%         
%         % test for unreshaping
%         dJUnresh = reshape(dJResh(1, :), [nUnitsRgn, nUnitsRgn]);
%         assert(isequal(squeeze(dJ(:, :, 1)), dJUnresh))
%         
%         for i = 1 : nD
%             
%             J_delta = squeeze(dJ(:, :, i)); % J_curr - J_prev;
%             
%             % rearrange to try to find potential structure in terms of
%             % changes
%             J_delta = J_delta(postSynSort, postSynSort); % makes sense to look at this since that's what gets changed....
%             
%             dJ_consecutive = [dJ_consecutive; J_delta(:)];
%             subplot(AxT(count)), imagesc(J_delta); axis tight
%             
%             pos = get(AxT(count), 'position');
%             pos(3) = 1 ./ (1.25 * nD);% 1.2 * pos(3); % adjust width
%             
%             if count > 1
%                 prev_pos = get(AxT(count - 1), 'position');
%             end
%             
%             if mod(count - 1, nD) == 0 || count == 1
%                 pos(1) = 0.2 * pos(1); % first plot of a row moves to L
%             else
%                 pos(1) = prev_pos(1) + prev_pos(3) + 0.2*pos(3); % adjust L position based on width of plots in earlier row
%             end
%             
%             set(AxT(count), 'position', pos)
%             
%             if count == 1
%                 presynLbl = text(AxT(count), round(nUnitsRgn/3), round(-1 * nUnitsRgn/12.5), 'pre-syn', 'fontsize', 10, 'fontweight', 'bold');
%                 postsynLbl = text(AxT(count), round(1.08*nUnitsRgn), round(nUnitsRgn/3), 'post-syn', 'fontsize', 10, 'fontweight', 'bold');
%                 set(postsynLbl, 'rotation', 270, 'horizontalalignment', 'left')
%             end
%             
%             if i == 1
%                 ylabel([rgnLabels{iRgn - 1}, ' (', num2str(nUnitsRgn), ' units)'], 'fontweight', 'bold', 'fontsize', 13)
%                 set(get(AxT(count), 'ylabel'), 'rotation', 90, 'horizontalalignment', 'center')
%             end
%             
%             if count >= length(AxT) - (nD) + 1
%                 xlabel([num2str(trlsToPlot(i) + 1), '\Delta', num2str(trlsToPlot(i))], 'fontsize', 12, 'fontweight', 'bold')
%                 if i == 1
%                     text(AxT(count), round(nUnitsRgn/5), round(1.25 * nUnitsRgn), '(\Delta from prev set)', 'fontsize', 10, 'fontweight', 'bold')
%                 end
%             end
%             
%             count = count + 1;
%             
%         end
%     end
%     
%     % update clims for the last three
%     [~, L, U] = isoutlier(dJ_consecutive, 'percentile', [0.5 99.5]);
%     
%     newCLims = [-1 * round(max(abs([L, U])), 3, 'decimals'), round(max(abs([L, U])), 3, 'decimals')];
%     set(AxT, 'clim', newCLims),
%     titleAxNum = round(0.5*(nD));
%     text(AxT(titleAxNum ), -95, -15, ...
%         [monkey, ssnDate,': consecutive within-set \DeltaJs --- ([0.5 99.5] %ile shown (range=\Delta +/-', num2str(newCLims(2), '%.2f'), ') --- SORTED ON: 1st \Delta (postsyn)'], 'fontweight', 'bold', 'fontsize', 14)
%     
%     set(figT, 'currentaxes', AxT),
%     print(figT, '-dtiff', '-r400', [RNNfigdir, 'intraRgnDeltaJs_consecutive', filesep, monkey, ssnDate, '_intraRgnDeltaJs_consecutive']),
%     close(figT)
    
    
     %% 2021-10-27 WIP : plot delta for intraregional J for consecutive samples from second trial
     
     activitySampleMat = NaN(sum(nUnitsAll), nSamples, nTrls);
     
     
     % Sloppy grabbing
     for iTrl = 1 : nTrls
         sampleTimePoints = allSPTimePoints(iTrl, :);
         % to save! note that J is reordered by region
         activitySampleAll{iTrl} = R_U{iTrl}(:, sampleTimePoints); % TO DO: FIGURE OUT WHAT THE ISSUE IS W INDEXING FROM MIN_LEN - SHOULDNT NEED TO TRANSFORM IT FOR PULLING MODEL ACTIVITY
         consecutiveJSampleAll{iTrl} = intraRgnConsJ{iTrl}; % 2021/11/14 shift to doing filtered matrix
     
         activitySampleMat(:, :, iTrl) = activitySampleAll{iTrl};
     end
     
     meanJAcrossTrls = mean(squeeze(cell2mat(arrayfun(@(iTrl) mean(consecutiveJSampleAll{iTrl}, [1 2]), 1 : nTrls, 'un', false))), 1);
     meanActAcrossTrls = mean(cell2mat(arrayfun(@(iTrl) mean(activitySampleAll{iTrl}, 1), 1 : nTrls, 'un', false)'), 1);
     stdJAcrossTrls = std(squeeze(cell2mat(arrayfun(@(iTrl) mean(consecutiveJSampleAll{iTrl}, [1 2]), 1 : nTrls, 'un', false))), 0, 1) ./ sqrt(sum(nUnitsAll));
     stdActAcrossTrls = std(cell2mat(arrayfun(@(iTrl) mean(activitySampleAll{iTrl}, 1), 1 : nTrls, 'un', false)'), 0, 1) ./ sqrt(sum(nUnitsAll));
     
     figure, errorbar(meanJAcrossTrls, stdJAcrossTrls, 'linewidth', 1.5), set(gca, 'fontweight', 'bold', 'fontsize', 12, 'xlim', [0.5 nSamples + 0.5]), xlabel('sample point#'), ylabel('pop mean J')
     saveas(gcf, [RNNfigdir, 'intraRgnDeltaJs_consecutiveSamplesSingleTrl', filesep, monkey, ssnDate, '_meanJAcrossTrls'], 'svg')
     % fig2svg([RNNfigdir, 'intraRgnDeltaJs_consecutiveSamplesSingleTrl', filesep, monkey, ssnDate, '_meanJAcrossTrls.svg'], ...
         % gcf, 1, [])
     print(gcf, '-dtiff', '-r400', [RNNfigdir, 'intraRgnDeltaJs_consecutiveSamplesSingleTrl', filesep, monkey, ssnDate, '_meanJAcrossTrls']),
     close
     
     figure, errorbar(meanActAcrossTrls, stdActAcrossTrls, 'linewidth', 1.5), set(gca, 'fontweight', 'bold', 'fontsize', 12, 'xlim', [0.5 nSamples + 0.5]), xlabel('sample point#'), ylabel('pop mean FR')
     % fig2svg([RNNfigdir, 'intraRgnDeltaJs_consecutiveSamplesSingleTrl', filesep, monkey, ssnDate, '_meanActAcrossTrls.svg'], ...
         % gcf, 1, [])
     print(gcf, '-dtiff', '-r400', [RNNfigdir, 'intraRgnDeltaJs_consecutiveSamplesSingleTrl', filesep, monkey, ssnDate, '_meanActAcrossTrls']),
     close
     
     for iSet = min(allSetIDs) : max(allSetIDs)
        activitySample = activitySampleAll(setID == iSet);
        consecutiveJSample = consecutiveJSampleAll(setID == iSet);
        consecutiveJSampleResh = intraRgnConsJResh(setID == iSet);
        
        trlIDsCurrSet = allTrialIDs(setID == iSet);
        
            % for the last trial of the first and last included sets, take the mean over all
            % samples to plot representative histograms
        if iSet == min(allSetIDs) 
            meanJOverLastTrlFirstSet = mean(consecutiveJSample{end}, 3);
            meanJOverLastTrlFirstSetPlot = reshape(sqrt(sum(nUnitsAll)) * meanJOverLastTrlFirstSet, sum(nUnitsAll)^2, 1);
        end
        
        if iSet == max(allSetIDs)
            meanJOverLastTrlLastSet = mean(consecutiveJSample{end}, 3);
            meanJOverLastTrlLastSetPlot = reshape(sqrt(sum(nUnitsAll)) * meanJOverLastTrlLastSet, sum(nUnitsAll)^2, 1);
        end
        
%         save([RNNSampleDir, 'classifier_matrices_', monkey, ssnDate, ...
%             '_set', num2str(iSet), '.mat'], 'activitySample', 'consecutiveJSample', 'trlIDsCurrSet', 'iSet', '-v7.3')
%     
%         save([RNNSampleDir, 'classifier_matrices_resh', monkey, ssnDate, ...
%             '_set', num2str(iSet), '.mat'], 'activitySample', 'consecutiveJSampleResh', 'trlIDsCurrSet', 'iSet', '-v7.3')
%     
    end
    
    % plot histograms of Js
    
    figure,
    [bincounts,edgesnew] = histcounts(meanJOverLastTrlFirstSetPlot, 75);
    histcenters = edgesnew(1:end-1) + (diff(edgesnew) ./ 2);
    % normalizing each by area under curve
    semilogy(histcenters,bincounts./sum(bincounts), 'o-', 'color', [0.12, 0.56, 1], 'linewidth', 1.5, 'markersize', 1.5)
    [bincounts,edgesnew] = histcounts(meanJOverLastTrlLastSetPlot, 75);
    histcenters = edgesnew(1:end-1) + (diff(edgesnew) ./ 2);
    hold on, semilogy(histcenters,bincounts./sum(bincounts), 'o-', 'color', [0 0 1], 'linewidth', 1.5, 'markersize', 1.5)
    set(gca, 'fontsize', 13, 'ylim', [0.0001 1], 'xlim', [-20 20])
    xlabel('Weight')
    ylabel('Density')
    lgd = legend({'first set', 'last set'}, 'location', 'northeast');
    print(gcf, '-dtiff', '-r400', [RNNfigdir, 'intraRgnDeltaJs_consecutiveSamplesSingleTrl', filesep, monkey, ssnDate, '_firstLastHists']),
    fig2svg([RNNfigdir, 'intraRgnDeltaJs_consecutiveSamplesSingleTrl', filesep, monkey, ssnDate, '_firstLastHists.svg'], ...
        gcf, 1, lgd)
    close     
           
    meanFRAcrossTrls = mean(cell2mat(arrayfun(@(iTrl) mean(activitySampleAll{iTrl}, 1), 1 : nTrls, 'un', false)'), 1);
    stdFRAcrossTrls = mean(cell2mat(arrayfun(@(iTrl) std(activitySampleAll{iTrl}, [], 1), 1 : nTrls, 'un', false)'), 1); % std over all units, averaged over all trials
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
        deltaIntraRgnConsJ = diff(cell2mat(arrayfun(@(iTrl) fullRgnConsJ{iTrl}(in_rgn, in_rgn, :), trlToPlot, 'un', 0)), 1, 3);
        

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
    %% PCA on consecutive intraregion Js 
    setToPlot = 2;
    
    tmpInds = find(trlInfo.trls_since_nov_stim == 0);
    
    while trlInfo.nov_stim_rwd_prob(tmpInds(setToPlot - 1)) == 999
        setToPlot = setToPlot + 1; % move down sets until the last one of prev trial is 999 (get it at most novel part of session
    end
    
    trlsToPlot = tmpInds(setToPlot - 1) : tmpInds(setToPlot) - 1; % # sets
    intraRgnConsJToPlot = intraRgnConsJ(trlsToPlot);% intraRgnJ(:, :, trlsToPlot);
    
    tmpJ = [];
    for iTrl = 1 : length(intraRgnConsJToPlot)
        tmpJ = cat(3, tmpJ, intraRgnConsJToPlot{iTrl});
    end
    
    % tmpJ =
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

        subJResh = cell2mat(arrayfun(@(iTrl) ...
            reshape(squeeze(subJ(:, :, iTrl)), 1, nUnitsRgn^2), 1 : size(subJ, 3), 'un', false)');
      
        subJReshAll = [subJReshAll, subJResh];
        
        % test for unreshaping
        assert(isequal(squeeze(subJ(:, :, 1)), reshape(subJResh(1, :), [nUnitsRgn, nUnitsRgn])))
        
        % threshold - place NaNs for J vals within cutoff
        % subJResh(:, rmElements) = NaN;
        
        % coeffs from all elements of J for each delta
        [w, scores, eigen, ~, pvar, mu] = pca(subJResh, 'Algorithm', 'svd', 'Centered', center_data, 'Economy', 0);
        
        wRgns{iRgn} = w;
        
        % project each regional subspace onto all data
        pjnRgns{iRgn} = (subJResh - repmat(mu,size(subJResh,1),1)) * w;
        % pjnRgns{iRgn} = subJResh * w;
        
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
        plot3(x(10 : 10 : end - 10, 1), x(10 : 10 : end - 10, 2), x(10 : 10 : end - 10, 3), 'linestyle', 'none', 'marker', 'o', 'markersize', 6, 'markeredgecolor', 'k', 'markerfacecolor', rgnColors(iRgn, :))
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
    
    %% full J a different way eff dim and low D pjn
%     [w, scores, eigen, ~, pvar, mu] = pca(subJReshAll, 'Algorithm', 'svd', 'Centered', center_data, 'Economy', 0);
%     
%     if ~isfolder([RNNfigdir, 'effDimFullIntraRgnJs_consecutive/'])
%         mkdir([RNNfigdir, 'effDimFullIntraRgnJs_consecutive/'])
%     end
%     
%     figure('color','w'); hold on,
%     set(gcf, 'units', 'normalized', 'outerposition', [0.2 0.15 0.6 0.7])
%     set(gca, 'fontweight', 'bold', 'fontsize', 13)
%     plot(1 : length(eigen), cumsum(pvar), 'linewidth', 1.5, 'color', 'b')
%     grid minor, xlabel('# PCs (1% of actual total)'), ylabel('% var explained')
%     xlim(gca, [0 0.01 * round(size(subJReshAll, 2))]), ylim(gca, [0 110])
%     nPC_cutoff = find(cumsum(pvar) >= 99,1, 'first');
%     line(gca, [nPC_cutoff nPC_cutoff], get(gca, 'ylim'), 'linestyle', '--', 'linewidth', 1.5, 'color', 'k')
%     text(gca, round(1.2 * nPC_cutoff), 10, ['#PCs = ', num2str(nPC_cutoff)], 'fontweight', 'bold')
%     title(gca, [monkey, ssnDate, ': eff dim (99% var exp) for consecutive Js (#trials=', num2str(size(subJReshAll, 1)), ')'], 'fontsize', 14, 'fontweight', 'bold')
%     print('-dtiff', '-r400', [RNNfigdir, 'effDimFullIntraRgnJs_consecutive', filesep, monkey, ssnDate, '_effDimFullIntraRgnJs_consecutive'])
%     close
%     
%     if ~isfolder([RNNfigdir, 'topPCsFullIntraRgnJs_consecutive/'])
%         mkdir([RNNfigdir, 'topPCsFullIntraRgnJs_consecutive/'])
%     end
%     
%     figure('color','w'); hold on,
%     set(gcf, 'units', 'normalized', 'outerposition', [0.2 0.15 0.6 0.7])
%     set(gca, 'fontweight', 'bold', 'fontsize', 13),
%     
%     nEl = [0; cumsum(nUnitsAll .^ 2)];
%     for iRgn = 2 : size(nEl)
%         startRgn = nEl(iRgn - 1) + 1;
%         stopRgn = nEl(iRgn) ;
%         
%         projData = subJReshAll(:, startRgn : stopRgn);
%         % tempProj = (projData - repmat(mu(startRgn : stopRgn), size(projData,1),1)) * w(startRgn : stopRgn, startRgn : stopRgn);
%         tempProj = (projData - repmat(mu, size(projData,1),1)) * w(startRgn : stopRgn, startRgn : stopRgn);
%         
%         x = tempProj;
%         all_lines(iRgn - 1) = plot3(x(:, 1), x(:, 2), x(:, 3), 'color', rgnColors(iRgn-1, :), 'linewidth', 1.5);
%         plot3(x(10 : 10 : end - 10, 1), x(10 : 10 : end - 10, 2), x(10 : 10 : end - 10, 3), 'linestyle', 'none', 'marker', 'o', 'markersize', 7, 'markeredgecolor', 'k', 'markerfacecolor', 'k')
%         plot3(x(1, 1), x(1, 2), x(1, 3), 'linestyle', 'none', 'marker', 'd', 'markersize', 10, 'markeredgecolor', 'k', 'markerfacecolor', 'g')
%         plot3(x(end, 1), x(end, 2), x(end, 3), 'linestyle', 'none', 'marker', 's', 'markersize', 10, 'markeredgecolor', 'k', 'markerfacecolor', 'r')
%         
%     end
%     
%     set(gca, 'xlim', [-xL xL], 'ylim', [-yL yL], 'zlim', [-zL zL])
%     grid minor
%     view(3)
%     xlabel('PC 1'), ylabel('PC 2'), zlabel('PC 3')
%     title(gca, [monkey, ssnDate, ': consecutive Js (#trials=', num2str(size(subJReshAll, 1)), ') projected into top 3 PCs'], 'fontsize', 14, 'fontweight', 'bold')
%     legend(all_lines, rgnLabels, 'location', 'bestoutside')
%     print('-dtiff', '-r400', [RNNfigdir, 'topPCsFullIntraRgnJs_consecutive', filesep, monkey, ssnDate, '_topPCsFullIntraRgnJs_consecutive'])
%     close
%     
    
end

