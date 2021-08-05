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
cd(mdlDir)
mdlFiles = dir('rnn_*_set*_trial*.mat');

allFiles = unique(arrayfun(@(i) ...
    mdlFiles(i).name(strfind(mdlFiles(i).name, 'rnn') + 4 : strfind(mdlFiles(i).name, 'set') - 2), 1:length(mdlFiles), 'un', false));

desiredOrder = {'left_cdLPFC', 'right_cdLPFC', ...
    'left_mdLPFC', 'right_mdLPFC', ...
    'left_vLPFC', 'right_vLPFC', ...
    'left_rdLPFC','right_rdLPFC'};

if setToPlot < 2
    disp('set # must be 2 or greater!')
    keyboard
end

% TO DO: TEST ON ALL OTHER SESSIONS
% TO DO: FIGURE OUT HOW TO MAKE MODEL OUTPUTS SMALLER - WHAT DO WE REALLY
% NEED? SAVE FULL ONES ELSEWHERE
for f = 2 : length(allFiles) % for each session....
    
    currSsn = dir(['rnn_', allFiles{f}, '_*.mat']);
    
    allSetIDs = unique(arrayfun(@(i) ...
        str2double(currSsn(i).name(strfind(currSsn(i).name,'set') + 3 : strfind(currSsn(i).name,'trial') - 2)), 1:length(currSsn)));
    allTrialIDs = unique(arrayfun(@(i) ...
        str2double(currSsn(i).name(strfind(currSsn(i).name,'trial') + 5 : end - 4)), 1:length(currSsn)));
    
    nTrls = length(allTrialIDs);
    nSets = length(allSetIDs);
    
    % for first trial of a session, pull params (this field is empty for
    % all other trials in a session)
    firstTrl = find(arrayfun(@(i) isequal(currSsn(i).name, ['rnn_', allFiles{f}, '_set0_trial1.mat']), 1 : nTrls));
    spikeInfoName = [currSsn(firstTrl).name(5 : median(strfind(currSsn(firstTrl).name, '_'))-1), '_meta.mat'];
    load([spikeInfoPath, spikeInfoName], 'spikeInfo', 'trlInfo', 'dsAllEvents')
    load([mdlDir, currSsn(firstTrl).name]);
    binSize = RNN.mdl.dtData; % in sec
    tDataInit = RNN.mdl.tData;
    tRNNInit = RNN.mdl.tRNN;
    
    % load full targets that went into fitCostaRNN for sanity checking with
    % targets from each trial's model
    metafnm = [spikeInfoPath, allFiles{f}, '_spikesCont'];
    S = load(metafnm);
    targets = S.dsAllSpikes;
    
    %% recreate target prep from fitCostaRNN for comparison of targets from each trial
    
    if RNN.mdl.params.doSmooth
        targets = smoothdata(targets, 2, 'movmean', 0.1 / RNN.mdl.dtData); % convert smoothing kernel from msec to #bins);
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
        % TO DO: CHECK REORDERING BY ITARGET
        
        % mdlfnm = [mdlDir, currSsn(i).name];
        mdlfnm = dir([mdlDir, 'rnn_', allFiles{f}, '_set*_trial', num2str(i), '.mat']);
        tmp = load(mdlfnm.name);
        
        mdl             = tmp.RNN.mdl;
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
        
        % clean up.mdl
        if isfield(mdl, 'tData')  && isfield(mdl, 'tRNN') && ~isequal(currSsn(i).name(end - 14 : end - 4), 'set0_trial1')
            mdl = rmfield(mdl, 'tData');
            mdl = rmfield(mdl, 'tRNN');
            RNN.mdl = mdl;
            save([mdlDir, mdlfnm.name], 'RNN')
        end
        
    end
    toc
    
    %% HOUSEKEEPING: SPATIAL
    
    % Ds = cell2mat(D_S');
    % Du = cell2mat(D_U');
    % figure, scatter(mean(Ds, 2), mean(cell2mat(R_dsU'), 2)) % shows these are correlated
    
    % find a way to turn Du into Ds order by minimizing sum of squared errors
    % between rows
    % uMatchs = arrayfun(@(n) ...
    %find(sum((Du - Ds(n, :)).^2, 2) == min(sum((Du - Ds(n, :)).^2, 2)), 1), 1 : size(Du, 1));
    % assert(isequaln(Du(uMatchs, :), Ds))
    % assert(sum([mean(Du(uMatchs, :), 2) -  mean(Ds, 2)]) < 1^-10)
    
    % TO DO: compare mean spike rates by region from spikeCount....or from their
    % figures on published data?
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
    minTrlsPerSet = min(trlsPerSetRNN(2 : end));
    DFull = cell2mat(D_U');
    
    % sanity check #1: make sure we are where we think we are in trlInfo
    % (for plotting consecutive trials (last trial prev set is first - to
    % compare delta between sets) as well as delta within set) using number
    % of trials using trial/set structure
    trl = trlInfo.trls_since_nov_stim;
    tmpLast = [find(trl == 0) - 1; height(trlInfo)];
    tmpLast(tmpLast == 0) = []; % delete first  trial
    trlsPerSetTbl = trl(tmpLast) + 1;
    n = numel(trlsPerSetRNN(2 : end));
    matchIx = find(arrayfun(@(i) isequal(trlsPerSetRNN(2 : end), trlsPerSetTbl(i : i + n - 1)), ...
        1 : numel(trlsPerSetTbl) - n)); % TO DO NEXT: USE THIS IN LATER PLOTTING!
    
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
    assert(best_lag == 0) % trial durations between trlInfo table and trlDurs from RNN snippets should match
    
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
    ylabel('pVar');
    set(gca,'Box','off','TickDir','out','FontSize',14);
    subplot(2,4,6);
    hold on;
    plot(firstChi2(1:nRun), 'k', 'linewidth', 1.5)
    ylabel('chi2');
    set(gca,'Box','off','TickDir','out','FontSize',14);
    print('-dtiff', '-r400', [RNNfigdir, allFiles{f}, '_firstTrlConvergence'])
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
    ylabel('pVar');
    set(gca,'Box','off','TickDir','out','FontSize',14);
    subplot(2,4,6);
    hold on;
    plot(lastChi2(1:nRun), 'k', 'linewidth', 1.5)
    ylabel('chi2');
    set(gca,'Box','off','TickDir','out','FontSize',14);
    print('-dtiff', '-r400', [RNNfigdir, allFiles{f}, '_lastTrlConvergence'])
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
    figure('color', 'w'), plot(1 : length(eigen), cumsum(pvar), 'linewidth', 1.5, 'color', 'b')
    set(gca, 'fontweight', 'bold', 'fontsize', 13),
    grid minor, xlabel('# PCs'), ylabel('% variance explained')
    nPC_cutoff = find(cumsum(pvar) >= 99,1, 'first');
    line(gca, [nPC_cutoff nPC_cutoff], get(gca, 'ylim'), 'linestyle', '--', 'linewidth', 1.5, 'color', 'k')
    title(gca, [allFiles{f}, ': eff dim (99% var exp) for avg activity'], 'fontsize', 14, 'fontweight', 'bold')
    print('-dtiff', '-r400', [RNNfigdir, allFiles{f}, '_effDimAllActivity'])
    close
    
    % project data into low D space
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
    title(gca, [allFiles{f}, ': avg activity from first # ', num2str(nTrls - sum(bad_trls)), ' trls projected into top 3 PCs'], 'fontsize', 14, 'fontweight', 'bold')
    print('-dtiff', '-r400', [RNNfigdir, allFiles{f}, '_topPCsAllActivity'])
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
        % deltaToSort = squeeze(dJ(:, :, end));
        deltaToSort = squeeze(dJ(:, :, 1)); % maybe more interesting since this is where the nov stim appears
        
        [~, preSynSort] = sort(mean(deltaToSort, 1), 'descend');
        [~, postSynSort] = sort(mean(deltaToSort, 2), 'descend');
        
        % reshape for PCA
        dJResh = cell2mat(arrayfun(@(iTrl) ...
            reshape(squeeze(dJ(:, :, iTrl)), 1, nUnitsRgn^2), 1 : nD, 'un', false)');
        
        % test for unreshaping
        dJUnresh = reshape(dJResh(1, :), [nUnitsRgn, nUnitsRgn]);
        isequal(squeeze(dJ(:, :, 1)), dJUnresh)
        
        
        for i = 1 : nD
            
            % J_prev = squeeze(intraRgnJ(in_rgn, in_rgn, trlsToPlot(i))); % trained J from first trial of current set
            % J_curr = squeeze(intraRgnJ(in_rgn, in_rgn, trlsToPlot(i) + 1)); % trained J from second trial of current set
            
            J_delta = squeeze(dJ(:, :, i)); % J_curr - J_prev;
            
            % rearrange to try to find potential structure in terms of
            % changes
            % J_delta = J_delta(preSynSort, preSynSort);
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
    [~, L, U, C] = isoutlier(dJ_consecutive, 'percentile', [0.5 99.5]);
    % newCLims =  [-0.45 0.45];
    
    % newCLims = [-1 * max(abs([L, U])), max(abs([L, U]))];
    newCLims = [-1 * round(max(abs([L, U])), 3, 'decimals'), round(max(abs([L, U])), 3, 'decimals')];
    set(AxT, 'clim', newCLims),
    titleAxNum = round(0.5*(nD));
    text(AxT(titleAxNum ), -95, -15, ...
        [allFiles{f},': consecutive within-set \DeltaJs --- ([0.5 99.5] %ile shown (range=\Delta +/-', num2str(newCLims(2), '%.2f'), ') --- SORTED ON: 1st \Delta (postsyn)'], 'fontweight', 'bold', 'fontsize', 14)
    
    set(figT, 'currentaxes', AxT),
    print(figT, '-dtiff', '-r400', [RNNfigdir, allFiles{f}, '_intraRgnDeltaJs_consecutiveTrls']),
    close(figT)
    
    
    %% PCA on delta for consecutive trials, intraregionally
    
    tmpInds = find(trlInfo.trls_since_nov_stim == 0);
    while trlInfo.nov_stim_rwd_prob(tmpInds(setToPlot - 1)) == 999
        setToPlot = setToPlot + 1; % move down sets until the last one of prev trial is 999 (get it at most novel part of session
    end
    
    trlsToPlot = tmpInds(setToPlot - 1) - 1 : nTrls; % for this one do as many as you have!
    
    subT = trlInfo(trlsToPlot, :); % to cross reference and make sure you're pulling Js from the trials you want

    dJ_consecutive = [];
    colormap(cm)
    
    for iRgn = 1 % find(~badRgn)

        in_rgn = newRgnInds(iRgn) + 1 : newRgnInds(iRgn + 1);
        nUnitsRgn = numel(in_rgn);
        assert(isequal(nUnitsRgn, nUnitsAll(iRgn)))
        
        % get difference matrix (changes in intraregional J) for desired trials
        dJ = diff(intraRgnJ(in_rgn, in_rgn, trlsToPlot), 1, 3);
        
        % reshape for PCA
        dJResh = cell2mat(arrayfun(@(iTrl) ...
            reshape(squeeze(dJ(:, :, iTrl)), 1, nUnitsRgn^2), 1 : size(dJ, 3), 'un', false)');
        
        % test for unreshaping
        dJUnresh = reshape(dJResh(1, :), [nUnitsRgn, nUnitsRgn]);
        assert(isequal(squeeze(dJ(:, :, 1)), dJUnresh))
        
        % coeffs from all elements of J for each delta
        [w, scores, eigen, ~, pvar, mu] = pca(dJResh, 'Algorithm', 'svd', 'Centered', center_data, 'Economy', 0);
        
        % estimate effective dimensionality
        figure('color', 'w'), plot(1 : length(eigen), cumsum(pvar), 'linewidth', 1.5, 'color', 'b')
        set(gca, 'fontweight', 'bold', 'fontsize', 13),
        grid minor, xlabel('# PCs'), ylabel('% variance explained')
        nPC_cutoff = find(cumsum(pvar) >= 99,1, 'first');
        line(gca, [nPC_cutoff nPC_cutoff], get(gca, 'ylim'), 'linestyle', '--', 'linewidth', 1.5, 'color', 'k')
        text(gca, round(1.2 * nPC_cutoff), 10, ['#PCs = ', num2str(nPC_cutoff)], 'fontweight', 'bold')
        title(gca, [allFiles{f}, ': eff dim (99% var exp) for consecutive \Delta Js (#=', num2str(size(dJ, 3)), ')'], 'fontsize', 14, 'fontweight', 'bold')
        print('-dtiff', '-r400', [RNNfigdir, allFiles{f}, '_effDimIntraRgnDeltaJs_consecutive'])
        close
        
        % project data into low D space
        figure('color', 'w'); hold on
        set(gcf, 'units', 'normalized', 'outerposition', [0.25 0.1 0.5 0.8])
        cm = colormap(cool);
        colormap(cm)
        
        % project delta Js back into low D space (is this valid since
        % like...it's not time?)
        projData = dJResh;
        tempProj = (projData - repmat(mu,size(projData,1),1)) * w;
        x = tempProj;
        plot3(x(:, 1), x(:, 2), x(:, 3), 'color', 'k', 'linewidth', 1.5)
        plot3(x(10 : 10 : end - 10, 1), x(10 : 10 : end - 10, 2), x(10 : 10 : end - 10, 3), 'linestyle', 'none', 'marker', 'o', 'markersize', 8, 'markeredgecolor', 'k', 'markerfacecolor', 'w')
        
        set(gca, 'fontweight', 'bold', 'fontsize', 13)
        xlabel('PC 1'), ylabel('PC 2'), zlabel('PC 3')
        grid minor
        view(3)
        title(gca, [allFiles{f}, ': consecutive \Delta Js (#=', num2str(size(dJ, 3)), ') projected into top 3 PCs'], 'fontsize', 14, 'fontweight', 'bold')
        print('-dtiff', '-r400', [RNNfigdir, allFiles{f}, '_topPCsIntraRgnDeltaJs_consecutive'])
        close
        
    end
    
    %% average target activity by region
    %
    %     % set up figure
    %         figure('color','w');
    %         set(gcf, 'units', 'normalized', 'outerposition', [0 0 1 1])
    %         AxTargs = arrayfun(@(i) subplot(4,2,i,'NextPlot', 'add', 'Box', 'on', 'TickDir','out', 'FontSize', 10, 'Fontweight', 'bold',  ...
    %             'xtick', 1:50:shortestTrl, 'xticklabel', dtData*((1:50:shortestTrl) - 1), 'ytick', '', 'ydir', 'reverse', 'xdir', 'normal'), 1:nRegions);
    %         cm = brewermap(100,'*RdGy');
    %
    %         softFac = 0.0000;
    %         for iRgn =  1 : nRegions
    %             if ~ismember(iRgn, find(badRgn))
    %                 rgnLabel = rgns{iRgn, 1};
    %                 rgnLabel(strfind(rgnLabel, '_')) = ' ';
    %
    %                 a = td.([rgns{iRgn,1}, '_spikes_avg']); % T x N
    %
    %                 % get the sort
    %                 [~, idx] = max(a, [], 1);
    %                 [~, rgnSort] = sort(idx);
    %
    %                 normfac = mean(abs(a), 1); % 1 x N
    %                 tmpRgn = a; %./ (normfac + softFac);
    %
    %                 subplot(AxTargs(iRgn))
    %                 imagesc(tmpRgn(:, rgnSort)' )
    %                 axis(AxTargs(iRgn), 'tight')
    %
    %                 title(rgnLabel, 'fontweight', 'bold')
    %                 set(gca, 'xcolor', rgnColors(iRgn,:),'ycolor', rgnColors(iRgn,:), 'linewidth', 2)
    %                 xlabel('time (sec)'), ylabel('neurons')
    %
    %                 if iRgn == 2
    %                     text(gca, -0.75*mean(get(gca,'xlim')), 1.1*max(get(gca,'ylim')), ...
    %                         [allFiles{f}, ' trial averaged target rates'], 'fontweight', 'bold', 'fontsize', 13)
    %                     [oldxlim, oldylim] = size(tmpRgn);
    %                 end
    %             end
    %         end
    %
    %         set(AxTargs, 'CLim', [0 1])
    %         oldpos = get(AxTargs(2),'Position');
    %         colorbar(AxTargs(2)), colormap(cm);
    %         set(AxTargs(2),'Position', oldpos)
    %         set(AxTargs(2), 'xlim', [0 oldxlim], 'ylim', [0 oldylim])
    %         print('-dtiff', '-r400', [RNNfigdir, allFiles{f}, '_trl_avged_targets_by_rgn'])
    %         close
    %
    
    
    %% initialize MP's data format (from all trials available)
    
    %     td = struct;
    %
    %     for iRgn = 1 : nRegions
    %         in_rgn = rgns{iRgn, 3};
    %         td.([rgns{iRgn,1}, '_spikes']) = DFull(in_rgn, :)';
    %         td.([rgns{iRgn,1}, '_spikes_avg']) = DMean(in_rgn, :)';
    %     end
    %
    %     %% compute currents, PCA on currents, and norm of top PCs for consecutive trials from a specified set, source, and target
    %
    %     for iTarget = 1:nRegions
    %         in_target = inArrays{iTarget};
    %
    %         for iSource = 1:nRegions
    %             in_source = inArrays{iSource};
    %
    %             if sum(in_target) >= n_PCs && sum(in_source) >= n_PCs
    %
    %                 P_inhAll = cell(nD, 1);
    %                 P_excAll = cell(nD, 1);
    %                 P_bothAll = cell(nD, 1);
    %
    %                 % get currents (normalized as in softNormalize as far as I can tell...)
    %                 for i = 1 : nD
    %
    %                     % For each trial, collect currents and combine together for PCA
    %                     JTrl = squeeze(J(:, :, trlsToPlot(i)));
    %                     RTrl = R_ds{trlsToPlot(i)};
    %
    %                     % compute all currents
    %                     P_both = JTrl(in_target, in_source) * RTrl(in_source, :);
    %
    %                     % compute inhibitory currents
    %                     J_inh = JTrl;
    %                     J_inh(J_inh > 0) = 0;
    %                     P_inh = J_inh(in_target, in_source) * RTrl(in_source, :);
    %
    %                     % compute excitatory currents
    %                     J_exc = JTrl;
    %                     J_exc(J_exc < 0) = 0;
    %                     P_exc = J_exc(in_target, in_source) * RTrl(in_source, :);
    %
    %                     P_bothAll{i} = P_both;
    %                     P_inhAll{i} = P_inh(:, :);
    %                     P_excAll{i} = P_exc(:, :);
    %
    %                 end
    %
    %                 % combine and transpose such that input is #samplepoints x #neurons
    %                 P_bothAll = cell2mat(P_bothAll')';
    %                 P_inhAll = cell2mat(P_inhAll')';
    %                 P_excAll = cell2mat(P_excAll')';
    %
    %                 if doSmooth
    %                     % inputs to smooth_data must be size [time channels]
    %                     P_bothAll = smooth_data(P_bothAll, binSize, binSize * 5)';
    %                     P_inhAll = smooth_data(P_inhAll, binSize, binSize * 5)';
    %                     P_excAll = smooth_data(P_excAll, binSize, binSize * 5)';
    %                 end
    %
    %                 % normalize after if smoothing by range of each neuron's firing
    %                 % rate (neurons x samplepoints)
    %                 P_bothAll = P_bothAll ./ (range(P_bothAll,2) + aFac);
    %                 P_inhAll = P_inhAll ./ (range(P_inhAll,2) + aFac);
    %                 P_excAll = P_excAll ./ (range(P_excAll,2) + aFac);
    %
    %                 % PCA on #samplepoints x #neurons for combined currents from
    %                 % all trials within a given source and region.
    %                 % w will be #neurons x #neurons
    %                 % scores = sp x neurons
    %                 % eigen = neurons x 1
    %                 % mu = 1 x neurons
    %                 [w, ~, ~, ~, ~, mu] = pca(P_bothAll', 'Algorithm','svd', 'Centered', center_data, 'Economy',0);
    %
    %                 % for each trial, combined currents only: project into low-D space and pick the num_dims requested
    %                 tempProjAll = cell(nD  + 1, 1);
    %                 normProjAll = cell(nD + 1, 1);
    %                 tmpInds = [0; cumsum(trlDurs(trlsToPlot))];
    %                 for i = 1 : nD
    %
    %                     % project over each trial
    %                     % tmpInds = trlStarts(trlsToPlot(i), 1) : trlStarts(trlsToPlot(i), 2);
    %                     iStart = tmpInds(i) + 1;
    %                     iStop = tmpInds(i + 1);
    %
    %                     projData = P_bothAll(:, iStart : iStop)'; % # sample points x neurons for a given stim
    %                     tempProj = (projData - repmat(mu,size(projData,1),1)) * w;
    %
    %                     % norm of n_PCs requested at each timepoint
    %                     top_proj = tempProj(:, 1:n_PCs);
    %                     normProj = zeros(size(top_proj,1),1);
    %                     for tt = 1:size(top_proj,1)
    %                         normProj(tt) = norm(top_proj(tt, :));
    %                     end
    %
    %                     tempProjAll{i} = tempProj;
    %                     normProjAll{i} = normProj;
    %
    %                 end
    %
    %                 td.(['Curr' rgns{iSource,1}, '_', rgns{iTarget,1}]) = P_bothAll'; % store as #samplepoints x #neurons
    %                 td.(['Curr' rgns{iSource,1}, '_', rgns{iTarget,1},'_inh']) = P_inhAll'; % store as #samplepoints x #neurons
    %                 td.(['Curr' rgns{iSource,1}, '_', rgns{iTarget,1},'_exc']) = P_excAll'; % store as #samplepoints x #neurons
    %                 td.(['Curr' rgns{iSource,1}, '_', rgns{iTarget,1}, '_pca']) = tempProjAll; % store each trial as #samplepoints x #neurons
    %                 td.(['Curr' rgns{iSource,1}, '_', rgns{iTarget,1}, '_pca_norm']) = normProjAll; % store each trial as #samplepoints x 1
    %             end
    %         end
    %     end
    %
    %     %% prune bad units from target activity and currents by cutting any unit whose overall FR is below threshold
    %
    %     for iSource = 1 : nRegions
    %         in_source = inArrays{iSource};
    %
    %         for iTarget = 1 : nRegions
    %             in_target = inArrays{iTarget};
    %
    %             if sum(in_target) >= n_PCs && sum(in_source) >= n_PCs
    %
    %                 Dtmp = td.([rgns{iTarget,1}, '_spikes']);
    %                 Dtmp_avg = td.([rgns{iTarget,1}, '_spikes_avg']);
    %                 bad_units = mean(Dtmp, 1) < 0.0005;
    %                 td.([rgns{iTarget,1}, '_spikes']) = Dtmp(:, ~bad_units);
    %                 td.([rgns{iTarget,1}, '_spikes_avg']) = Dtmp_avg(:, ~bad_units);
    %
    %                 % remove target units below FR threshold from currents
    %                 td.(['Curr' rgns{iSource,1} '_', rgns{iTarget,1}]) = td.(['Curr' rgns{iSource,1} '_', rgns{iTarget,1}])(:, ~bad_units);
    %             end
    %         end
    %     end
    %
    %     %% plot currents (imagesc) - to do: currents over time or currents averaged?
    %
    %     close all
    %     cm = brewermap(100,'*RdBu');
    %     allP = [];
    %
    %     figure('color','w');
    %     set(gcf, 'units', 'normalized', 'outerposition', [0.05 0.05 0.9 0.9])
    %
    %     AxCurr = arrayfun(@(i) subplot(4,2,i,'NextPlot', 'add', 'Box', 'off',  'TickDir', 'out', 'FontSize', 10, 'Fontweight', 'bold',...
    %         'xtick', newTrlInds(1 : nD + 1), 'xticklabel', 1 : nD + 1, 'ytick', '', 'ydir', 'reverse', 'xdir', 'normal'), 1:nRegions);
    %     count = 1;
    %
    %     for iTarget = 1:nRegions % One plot per target
    %         in_target = inArrays{iTarget};
    %
    %         for iSource = 1:nRegions
    %
    %             if iSource ~= iTarget
    %                 continue
    %             end
    %
    %             if ~ismember(iTarget, find(badRgn))
    %                 a = td.(['Curr', rgns{iTarget,1}, '_', rgns{iTarget,1}]);
    %                 a = a(1 : newTrlInds(2) - 1, :); % get sort from first trial
    %                 a = a./repmat(mean(abs(a), 1), size(a, 1), 1);
    %                 idx = 1 : size(a, 2);
    %             else
    %                 continue
    %             end
    %
    %             in_source = inArrays{iSource};
    %
    %             if sum(in_target) >= n_PCs && sum(in_source) >= n_PCs
    %
    %                 subplot(AxCurr(iSource));
    %
    %                 % divide by mean(abs(val)) of current for each unit
    %                 P = td.(['Curr', rgns{iSource,1}, '_', rgns{iTarget,1}]);
    %                 P = P(1 : newTrlInds(nD + 2) - 1, :); % do first nD + 1 trls
    %                 P = P(:,idx); P = P ./ mean(abs(P),1);
    %
    %                 allP = [allP; P(:)];
    %
    %                 imagesc(P'); axis tight
    %
    %                 for s = 2 : nD + 1
    %                     line(gca, [newTrlInds(s), newTrlInds(s)], get(gca, 'ylim'), 'color', 'black', 'linewidth', 1.25)
    %                 end
    %
    %                 title([rgns{iSource,1} ' > ' rgns{iTarget,1}],'FontWeight', 'bold')
    %                 xlabel('trials'), ylabel('neurons')
    %
    %                 if count==2
    %                     text(AxCurr(count), -(1.25) * mean(get(AxCurr(count),'xlim')), -25, ...
    %                         [allFiles{f}, ': intra-rgn currents over 1st ', num2str(nD + 1), ' trls'], 'fontweight', 'bold', 'fontsize', 13)
    %                 end
    %
    %                 if iSource == 2
    %                     [oldxlim, oldylim] = size(P);
    %                 end
    %
    %             end
    %
    %             count = count + 1;
    %
    %         end
    %
    %     end
    %
    %     [~, L, U] = isoutlier(allP, 'percentile', [0.5 99.5]);
    %     c = [-1 * max(abs([L, U])), max(abs([L, U]))];
    %     set(AxCurr, 'clim', c)
    %     oldpos = get(AxCurr(2),'Position');
    %     colorbar(AxCurr(2)), colormap(cm);
    %     set(AxCurr(2),'Position', oldpos)
    %     set(AxCurr(2), 'xlim', [0 oldxlim], 'ylim', [0 oldylim])
    %
    %     print('-dtiff', '-r400', [RNNfigdir, allFiles{f}, '_intraRgnCurrents_consecutiveTrls'])
    %
    
end

