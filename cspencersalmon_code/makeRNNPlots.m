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
setToPlot       = 1; % only one for now

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


for f = 3 : length(allFiles) % for each session....
    
    currSsn = dir(['rnn_', allFiles{f}, '_*.mat']);
    
    allSetIDs = unique(arrayfun(@(i) ...
        str2double(currSsn(i).name(strfind(currSsn(i).name,'set') + 3 : strfind(currSsn(i).name,'trial') - 2)), 1:length(currSsn)));
    allTrialIDs = unique(arrayfun(@(i) ...
        str2double(currSsn(i).name(strfind(currSsn(i).name,'trial') + 5 : end - 4)), 1:length(currSsn)));
    
    nTrls = length(allTrialIDs);
    nSets = length(allSetIDs);
    
    % plot every xth trl or set since too many
    xthTrl = round((nTrls - 1) / nDeltasToPlot);
    xthSet = round((nSets - 1) / nDeltasToPlot);
    
    J = [];
    J0 = [];
    D = cell(nTrls, 1);
    R_ds = cell(nTrls, 1);
    tData = cell(nTrls, 1);
    setID = [];
    trlNum = [];
    
    % for first trial of a session, pull params (this field is empty for
    % all other trials in a session)
    firstTrl = find(arrayfun(@(i) isequal(currSsn(i).name, ['rnn_', allFiles{f}, '_set0_trial1.mat']), 1 : nTrls));
    spikeInfoName = [currSsn(firstTrl).name(5 : median(strfind(currSsn(firstTrl).name, '_'))-1), '_meta.mat'];
    load([spikeInfoPath, spikeInfoName], 'spikeInfo', 'trlInfo', 'dsAllEvents')
    load([mdlDir, currSsn(firstTrl).name]);
    binSize = RNN.mdl.dtData; % in sec
    
    % set up indexing vectors for submatrices
    rgns            = RNN.mdl.params.arrayRegions;
    dtData          = RNN.mdl.dtData;
    arrayList       = rgns(:, 2);
    nRegions        = length(arrayList);
    rgnColors       = brewermap(nRegions, 'Spectral');% cmap(round(linspace(1, 255, nRegions)),:);
    
    
    % collect outputs for all trials in a session
    tic
    parfor i = 1 : nTrls
        
        mdlfnm = [mdlDir, currSsn(i).name];
        tmp = load(mdlfnm);
        
        mdl             = tmp.RNN.mdl;
        J(:, :, i)      = mdl.J;
        J0(:, :, i)     = mdl.J0;
        D{i}            = mdl.targets;
        setID(i)        = mdl.setID;
        trlNum(i)       = mdl.iTrl;
        
        % get indices for each sample of model data
        tData{i}        = mdl.tData;
        R_ds{i}         = mdl.RMdlSample;
    end
    toc
    
    %% HOUSEKEEPING: SPATIAL
    
    % make regions more legible
    rgnOrder = arrayfun(@(iRgn) find(strcmp(rgns(:,1), desiredOrder{iRgn})), 1 : nRegions); % ensure same order across sessions
    rgns(:, 1) = arrayfun(@(iRgn) strrep(rgns{iRgn, 1}, 'left_', 'L'), 1 : nRegions, 'un', false)';
    rgns(:, 1) = arrayfun(@(iRgn) strrep(rgns{iRgn, 1}, 'right_', 'R'), 1 : nRegions, 'un', false)';
    rgns = rgns(rgnOrder, :);
    rgnLabels = rgns(:, 1);
    rgnLabels = arrayfun(@(iRgn) rgnLabels{iRgn}(1:end-3), 1:nRegions, 'un', false);
    rgnIxToPlot = cell2mat(arrayfun(@(iRgn) ~isempty(strfind(rgnLabels{iRgn}(1), 'L')), 1 : nRegions, 'un', false));
    
    % id bad rgns
    inArrays = rgns(:, 3);
    % rgn w really low # units gets excluded
    badRgn = arrayfun(@(iRgn) sum(inArrays{iRgn}) < n_PCs, 1:nRegions);
    
    %% HOUSEKEEPING: TEMPORAL
   
    % reorder by ascending trlNum
    [~, trlSort] = sort(trlNum, 'ascend');
    % quick check to make sure no consecutive trials are skipped!
    assert(numel(unique(diff(trlNum(trlSort)))) == 1)
    J = J(:, :, trlSort);
    J0 = J0(:, :, trlSort);
    D = D(trlSort);
    setID = setID(trlSort);
    tData = tData(trlSort);
    R_ds = R_ds(trlSort);
    
    % for demarcating new trials in later plotting
    DFull = cell2mat(D');
    nSPFull = size(DFull,2);
    nSPTmp = cell2mat(cellfun(@size, D, 'un', 0));
    nSP = nSPTmp(:, 2);
    newTrlInds = [1; cumsum(nSP(1:end-1))+1];
    trlStartsTmp = [newTrlInds - 1; size(DFull, 2)];
    trlStarts = cell2mat(arrayfun(@(iTrl) [trlStartsTmp(iTrl, 1)+1 trlStartsTmp(iTrl+1, 1)], 1:nTrls, 'un', 0)');
    
    % get the largest possible minimum number of trials from each set
    [~, ia, ic] = unique(setID');
    a_counts = accumarray(ic, 1);
    minTrlsPerSet = min(a_counts(2 : end));
    
    % for plotting consecutive trials
    trlsToPlot = find(setID == setToPlot);
    
    % if set nD is greater than #trls in set, reassign nD according to min
    % # trls per set over all available sets
    if numel(trlsToPlot) < (nD + 1) || isempty(nD)
        nD = minTrlsPerSet - 1;
    end
    
    trlsToPlot = trlsToPlot(1 : nD + 1);
    
    % for trial averaging from all available trials (get the shortest trial and
    % then trim data from each trial to be as long as that one, then take
    % average over all those trials (for projecting onto)
    trlDurs = arrayfun(@(iTrl) size(D{iTrl}, 2), 1 : nTrls)';
    shortestTrl = min(trlDurs(:));
    DTrunc = D(:);
    DTrunc = arrayfun(@(iTrl) DTrunc{iTrl}(:, 1 : shortestTrl), 1 : nD + 1, 'un', false)';
    DTrunc = cell2mat(DTrunc');
    DMean = cell2mat(arrayfun(@(n) mean(reshape(DTrunc(n, :), shortestTrl, size(DTrunc, 2)/shortestTrl), 2)', 1 : size(DTrunc, 1), 'un', false)');
    
    % for demarcating sets
    firstTrlNewSet = cumsum(arrayfun(@(iSet) sum(setID == allSetIDs(iSet)), 1:length(allSetIDs)));
    firstTrlNewSet = firstTrlNewSet(1 : end - 1) + 1;
    lastTrlEachSet = cumsum(arrayfun(@(iSet) sum(setID == allSetIDs(iSet)), 1:length(allSetIDs)));
    lastTrlEachSet = lastTrlEachSet(2:end);

    % for grabbing necessary event info
    fixOnInds = find(dsAllEvents == 1);
    trlIx = fixOnInds(1) : fixOnInds(nTrls + 1) - 1;  % from fixOn first trial to right before fixOn of next trial 
    allEvents = dsAllEvents(trlIx);
    assert(size(allEvents, 2) == size(DFull, 2))
    assert(sum(allEvents == 1) == nTrls)
    trlInfo = trlInfo(1 : nTrls, :);
    
    durBtwnTrls = [NaN; diff(trlInfo.event_times(:, 1))];
    
    trlsPerSet = histcounts(setID, [-0.5 : 1 : max(setID) + 0.5]);
    minTrlsPerSet = min(trlsPerSet(2 : end));
    
    %% reorder J by region order (for J plotting of full matrix)
    
    Jordr = zeros(size(J));
    count = 1;
    nUnitsAll = NaN(nRegions, 1);
    newOrder = [];
    
    for iRgn = 1 : nRegions
        in_rgn = rgns{iRgn,3};
        newOrder = [newOrder; find(in_rgn)]; % reorder J so that rgns occur in order
        nUnitsRgn = sum(in_rgn);
        nUnitsAll(iRgn) = nUnitsRgn;
        newIdx = count : count + nUnitsRgn - 1;
        Jordr(newIdx, newIdx, :) = J(in_rgn, in_rgn, :); % intra-region only
        count = count + nUnitsRgn;
    end
    
    Jtmp2 = J(newOrder, newOrder, :); % includes interactions
    
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
    
    %% set up for future J plots reordered so within-rgn is on-diagonal and between-rgn is off-diagonal
    
    tmp = [0; nUnitsAll];
    JLblPos = arrayfun(@(iRgn) sum(tmp(1 : iRgn-1)) + tmp(iRgn)/2, 2 : nRegions); % for the labels separating submatrices
    JLinePos = cumsum(nUnitsAll(~badRgn))'; % for the lines separating regions
    newRgnInds = [0, JLinePos];
    rgnLabels = rgnLabels(~badRgn);
    nUnitsAll = nUnitsAll(~badRgn);
    rgnIxToPlot = rgnIxToPlot(~badRgn);
    
    %% initialize MP's data format (from all trials available)
    
    td = struct;
    
    for iRgn = 1 : nRegions
        in_rgn = rgns{iRgn, 3};
        td.([rgns{iRgn,1}, '_spikes']) = DFull(in_rgn, :)';
        td.([rgns{iRgn,1}, '_spikes_avg']) = DMean(in_rgn, :)';
    end
    
    %% compute currents, PCA on currents, and norm of top PCs for consecutive trials from a specified set, source, and target
    
    for iTarget = 1:nRegions
        in_target = inArrays{iTarget};
        
        for iSource = 1:nRegions
            in_source = inArrays{iSource};
            
            if sum(in_target) >= n_PCs && sum(in_source) >= n_PCs
                
                P_inhAll = cell(nD + 1, 1);
                P_excAll = cell(nD + 1, 1);
                P_bothAll = cell(nD + 1, 1);
                
                % get currents (normalized as in softNormalize as far as I can tell...)
                for i = 1 : nD + 1
                    
                    % For each trial, collect currents and combine together for PCA
                    JTrl = squeeze(J(:, :, trlsToPlot(i)));
                    RTrl = R_ds{trlsToPlot(i)};
                    
                    % compute all currents
                    P_both = JTrl(in_target, in_source) * RTrl(in_source, :);
                    
                    % compute inhibitory currents
                    J_inh = JTrl;
                    J_inh(J_inh > 0) = 0;
                    P_inh = J_inh(in_target, in_source) * RTrl(in_source, :);
                    
                    % compute excitatory currents
                    J_exc = JTrl;
                    J_exc(J_exc < 0) = 0;
                    P_exc = J_exc(in_target, in_source) * RTrl(in_source, :);
                    
                    P_bothAll{i} = P_both;
                    P_inhAll{i} = P_inh(:, :);
                    P_excAll{i} = P_exc(:, :);
                    
                end
                
                % combine and transpose such that input is #samplepoints x #neurons
                P_bothAll = cell2mat(P_bothAll')';
                P_inhAll = cell2mat(P_inhAll')';
                P_excAll = cell2mat(P_excAll')';
                
                if doSmooth
                    % inputs to smooth_data must be size [time channels]
                    P_bothAll = smooth_data(P_bothAll, binSize, binSize * 5)';
                    P_inhAll = smooth_data(P_inhAll, binSize, binSize * 5)';
                    P_excAll = smooth_data(P_excAll, binSize, binSize * 5)';
                end
                
                % normalize after if smoothing by range of each neuron's firing
                % rate (neurons x samplepoints)
                P_bothAll = P_bothAll ./ (range(P_bothAll,2) + aFac);
                P_inhAll = P_inhAll ./ (range(P_inhAll,2) + aFac);
                P_excAll = P_excAll ./ (range(P_excAll,2) + aFac);
                
                % PCA on #samplepoints x #neurons for combined currents from
                % all trials within a given source and region.
                % w will be #neurons x #neurons
                % scores = sp x neurons
                % eigen = neurons x 1
                % mu = 1 x neurons
                [w, ~, ~, ~, ~, mu] = pca(P_bothAll', 'Algorithm','svd', 'Centered', center_data, 'Economy',0);
                
                % for each trial, combined currents only: project into low-D space and pick the num_dims requested
                tempProjAll = cell(nD  + 1, 1);
                normProjAll = cell(nD + 1, 1);
                tmpInds = [0; cumsum(trlDurs(trlsToPlot))];
                for i = 1 : nD + 1
                    
                    % project over each trial
                    % tmpInds = trlStarts(trlsToPlot(i), 1) : trlStarts(trlsToPlot(i), 2);
                    iStart = tmpInds(i) + 1;
                    iStop = tmpInds(i + 1);
                    
                    projData = P_bothAll(:, iStart : iStop)'; % # sample points x neurons for a given stim
                    tempProj = (projData - repmat(mu,size(projData,1),1)) * w;
                    
                    % norm of n_PCs requested at each timepoint
                    top_proj = tempProj(:, 1:n_PCs);
                    normProj = zeros(size(top_proj,1),1);
                    for tt = 1:size(top_proj,1)
                        normProj(tt) = norm(top_proj(tt, :));
                    end
                    
                    tempProjAll{i} = tempProj;
                    normProjAll{i} = normProj;
                    
                end
                
                td.(['Curr' rgns{iSource,1}, '_', rgns{iTarget,1}]) = P_bothAll'; % store as #samplepoints x #neurons
                td.(['Curr' rgns{iSource,1}, '_', rgns{iTarget,1},'_inh']) = P_inhAll'; % store as #samplepoints x #neurons
                td.(['Curr' rgns{iSource,1}, '_', rgns{iTarget,1},'_exc']) = P_excAll'; % store as #samplepoints x #neurons
                td.(['Curr' rgns{iSource,1}, '_', rgns{iTarget,1}, '_pca']) = tempProjAll; % store each trial as #samplepoints x #neurons
                td.(['Curr' rgns{iSource,1}, '_', rgns{iTarget,1}, '_pca_norm']) = normProjAll; % store each trial as #samplepoints x 1
            end
        end
    end
    
    %% WIP: pca the J space???
    
    %     for iTarget = 1:nRegions
    %         in_target = inArrays{iTarget};
    %         nTarg = sum(in_target);
    %         for iSource = 1:nRegions
    %             in_source = inArrays{iSource};
    %             nSource = sum(in_source);
    %             if nTarg >= n_PCs && nSource >= n_PCs
    %                 subJResh =  cell2mat(arrayfun(@(iTrl) ...
    %                     reshape(squeeze(J(in_target, in_source, iTrl)), 1, nTarg * nSource), 1 : nTrls, 'un', false)');
    %                 % test for un reshaping
    %                 % JUnresh = reshape(subJResh(1,:), [nTarg, nSource]);
    %                 % isequal(squeeze(J(in_target, in_source, 1)), JUnresh)
    %
    %                 % arrayfun(@(iSet) subJResh(firstTrlNewSet(iSet) : firstTrlNewSet(iSet) + minTrlsPerSet - 1, :), 1 : nSets - 1, 'un', false)
    %
    %                 % PCA on #trials x #cxns for Js from
    %                 % all trials within a given source and region.
    %                 % w will be #cxn x #cxn
    %                 % scores = trials x cxn
    %                 % eigen = cxn x 1
    %                 % mu = 1 x cxn
    %                 subJResh = subJResh';
    %
    %                 % quick aside to make tensor for TCA
    %                 T_tmp = arrayfun(@(iSet) subJResh(:, firstTrlNewSet(iSet) : firstTrlNewSet(iSet) + minTrlsPerSet - 1), 1 : nSets - 1, 'un', false);
    %                 T = NaN(nTarg * nSource, minTrlsPerSet, nSets - 1);
    %
    %                 for iSet = 1 : nSets - 1
    %                     T(:, :, iSet) = T_tmp{iSet};
    %                 end
    %
    %                 subJResh = subJResh ./ (range(subJResh,2)); % "balance desire for PCA to explain all cxns with desire that weak cxns not contribute equally to strong ones"
    %                 subJResh = subJResh';
    %                 [w, scores, eigen, ~, pvar, mu] = pca(subJResh, 'Algorithm', 'svd', 'Centered', center_data, 'Economy', 0);
    %                 n_PCs = find(cumsum(pvar)>=95,1); % gives you elements covering up to
    %                 % 95% variance
    %
    %                 % for each trial,  project into low-D space and pick the num_dims requested
    %                     % project over each trial
    %                     projData = subJResh; % # sample points x #cxns for a given stim
    %                     tempProj = (projData - repmat(mu,size(projData,1),1)) * w;
    %
    %                     % norm of n_PCs requested at each timepoint
    %                     top_proj = tempProj(:, 1:n_PCs);
    %                     normProj = zeros(size(top_proj,1),1);
    %                     for tt = 1:size(top_proj,1)
    %                         normProj(tt) = norm(top_proj(tt, :));
    %                     end
    %
    %
    %             end
    %
    %
    % %             figure, plot(normProj)
    % %             hold on
    % %             for s = 1:length(firstTrlNewSet)
    % %                 line(gca, [firstTrlNewSet(s), firstTrlNewSet(s)], get(gca, 'ylim'), 'color', 'black', 'linewidth', 1.5)
    % %             end
    %
    %         end
    %     end
    
    
    %% prune bad units from target activity and currents by cutting any unit whose overall FR is below threshold
    
    for iSource = 1 : nRegions
        in_source = inArrays{iSource};
        
        for iTarget = 1 : nRegions
            in_target = inArrays{iTarget};
            
            if sum(in_target) >= n_PCs && sum(in_source) >= n_PCs
                
                Dtmp = td.([rgns{iTarget,1}, '_spikes']);
                Dtmp_avg = td.([rgns{iTarget,1}, '_spikes_avg']);
                bad_units = mean(Dtmp, 1) < 0.0005;
                td.([rgns{iTarget,1}, '_spikes']) = Dtmp(:, ~bad_units);
                td.([rgns{iTarget,1}, '_spikes_avg']) = Dtmp_avg(:, ~bad_units);
                
                % remove target units below FR threshold from currents
                td.(['Curr' rgns{iSource,1} '_', rgns{iTarget,1}]) = td.(['Curr' rgns{iSource,1} '_', rgns{iTarget,1}])(:, ~bad_units);
            end
        end
    end
    
    %% WIP: PCA of activity over nD + 1 trials from chosen set idk look up jPCA trajectories look up difference between   
    % TO DO: REMEMBER HOW AND WHERE WE GET THE CONTINUOUSNESS - MAYBE ONLY
    % PCA UP TO FIX OFF?
    % remove bad units from  activity
    d = cell2mat(D');
    [~, FR_cutoff] = isoutlier(mean(d, 2), 'percentile', [0.5 100]); % [0.5 100] % throw out outliers since not good for PCA
    bad_units = mean(d, 2) <= FR_cutoff;
   
    % throw out any trials that are weirdly long or short
    trlDurs = arrayfun(@(s) size(D{s}, 2), 1 : size(D, 1));
    [~, minTrlLen, maxTrlLen] = isoutlier(trlDurs, 'percentile', [5 95]); % [1 99]
    bad_trls = trlDurs >= maxTrlLen | trlDurs <= minTrlLen; 
    Dtmp = D(~bad_trls);
    T = trlInfo(~bad_trls, :);
    durBtwnTrls = durBtwnTrls(~bad_trls, :);
    longestTrl = max(arrayfun(@(s) max(size(Dtmp{s}, 2)), 1 : size(Dtmp, 1)));
    
    % elapsed time between events (1 = fixation, 2 = stim, 3 = choice, 4 = outcome, 5 = time of next trl
    % fixation)
    avgEventTS = [mean(T.aligned_event_times, 1), mean(durBtwnTrls)];
    eventInds = round(avgEventTS / dtData); 
    % pad remaining units and trials to same length
    trlsPadded = arrayfun(@(s) ...
        [Dtmp{s}, NaN(size(Dtmp{s}, 1), longestTrl - size(Dtmp{s}, 2))], 1 : size(Dtmp, 1), 'un', false);
    X = cell2mat(trlsPadded);
    
    % trial average them 
    trlsAvg = cell2mat(arrayfun( @(n) nanmean(reshape(X(n, :), longestTrl, (nTrls - sum(bad_trls))), 2), 1 : size(X, 1), 'un', false))';
    trlsAvg(bad_units, :) = [];
    
    % coeffs from all activity, trial averaged and padded
    [w, scores, eigen, ~, pvar, mu] = pca(trlsAvg', 'Algorithm', 'svd', 'Centered', center_data, 'Economy', 0);
    
    % project data into low D space
    figure('color','w'); hold on
    set(gcf, 'units', 'normalized', 'outerposition', [0.25 0.1 0.5 0.8])
    cm = colormap(cool);
    colormap(cm)
    
    % project padded trial averaged activity (all) into low D space
    projData = trlsAvg'; 
    tempProj = (projData - repmat(mu,size(projData,1),1)) * w;
    x = tempProj;
    plot3(x(1, 1), x(1, 2), x(1, 3), 'color', 'g', 'marker', '^', 'markerfacecolor', 'g', 'markeredgecolor', 'g', 'markersize', 15)
    plot3(x(end, 1), x(end, 2), x(end, 3), 'color', 'r', 'marker', 's', 'markerfacecolor', 'r', 'markeredgecolor', 'r', 'markersize', 15)
    plot3(x(:, 1), x(:, 2), x(:, 3), 'color', 'k', 'linewidth', 1.5)
    plot3(x(10 : 10 : end - 10, 1), x(10 : 10 : end - 10, 2), x(10 : 10 : end - 10, 3), 'linestyle', 'none', 'marker', 'o', 'markersize', 10, 'markerfacecolor', 'k')
    
    % project padded average of first trials of each new set into low D
    % space 
    % TO DO: WIP
%     firstTrls = D(firstTrlNewSet);
%     firstTrlDurs = arrayfun(@(s) size(firstTrls{s}, 2), 1 : size(firstTrls, 1)); 
%     bad_first_trls = firstTrlDurs >= maxTrlLen | firstTrlDurs <= minTrlLen; % use same cut off as for full dataset
%     firstTrls = firstTrls(~bad_first_trls);
%     firstTrlsPadded = arrayfun(@(s) ...
%         [firstTrls{s}, NaN(size(firstTrls{s}, 1), longestTrl - size(firstTrls{s}, 2))], 1 : size(firstTrls, 1), 'un', false);
%     X = cell2mat(firstTrlsPadded);
%     firstTrlsAvg = cell2mat(arrayfun(@(n) nanmean(reshape(X(n, :), size(X, 2) / (nSets - 1), (nSets - 1)), 2), 1 : size(X, 1), 'un', false))';
%     projData = firstTrlsAvg(~bad_units, :)'; %
%     tempProj = (projData - repmat(mu,size(projData,1),1)) * w;
%     x = tempProj;
%     plot3(x(1, 1), x(1, 2), x(1, 3), 'color', cm(1, :), 'marker', '^', 'markerfacecolor', cm(1, :), 'markersize', 15)
%     plot3(x(end, 1), x(end, 2), x(end, 3), 'color', cm(1, :), 'marker', 's', 'markerfacecolor', cm(1, :), 'markersize', 15)
%     plot3(x(:, 1), x(:, 2), x(:, 3), 'color', cm(1, :), 'linewidth', 1.5)
%     
%     
    % project padded average of last trials of each new set into low D
    % space
%     allLastTrls = D(lastTrlEachSet);
%     allLastTrlsPadded = arrayfun(@(s) ...
%         [allLastTrls{s}, NaN(size(allLastTrls{s}, 1), longestTrl - size(allLastTrls{s}, 2))], 1 : size(allLastTrls, 1), 'un', false);
%     X = cell2mat(allLastTrlsPadded);
%     lastTrlsAvg = cell2mat(arrayfun(@(n) nanmean(reshape(X(n, :), size(X, 2) / (nSets - 1), (nSets - 1)), 2), 1 : size(X, 1), 'un', false))';
    % project consecutive activity into low D space using weight matrix from above
%     projData = lastTrlsAvg(~bad_units, :)'; %
%     tempProj = (projData - repmat(mu,size(projData,1),1)) * w;
%     x = tempProj;
%     plot3(x(1, 1), x(1, 2), x(1, 3), 'color', cm(end, :), 'marker', '^', 'markerfacecolor', cm(end, :), 'markersize', 15)
%     plot3(x(end, 1), x(end, 2), x(end, 3), 'color', cm(end, :), 'marker', 's', 'markerfacecolor', cm(end, :), 'markersize', 15)
%     plot3(x(:, 1), x(:, 2), x(:, 3), 'color', cm(end, :), 'linewidth', 1.5)
    
    set(gca, 'fontweight', 'bold', 'fontsize', 13)
    xlabel('PC 1'), ylabel('PC 2'), zlabel('PC 3')
    grid minor
    view(3)
    title(gca, [allFiles{f}, ': avg activity from set #', num2str(setToPlot), ' (# ', num2str(nD+1), ' trls) projected into top 3 PCs'], 'fontsize', 14, 'fontweight', 'bold')
    print('-dtiff', '-r400', [RNNfigdir, allFiles{f}, '_topPCsAllActivity'])
    close
    
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
    %% plot currents (imagesc) - to do: currents over time or currents averaged?
    
    close all
    cm = brewermap(100,'*RdBu');
    allP = [];
    
    figure('color','w');
    set(gcf, 'units', 'normalized', 'outerposition', [0.05 0.05 0.9 0.9])
    
    AxCurr = arrayfun(@(i) subplot(4,2,i,'NextPlot', 'add', 'Box', 'off',  'TickDir', 'out', 'FontSize', 10, 'Fontweight', 'bold',...
        'xtick', newTrlInds(1 : nD + 1), 'xticklabel', 1 : nD + 1, 'ytick', '', 'ydir', 'reverse', 'xdir', 'normal'), 1:nRegions);
    count = 1;
    
    for iTarget = 1:nRegions % One plot per target
        in_target = inArrays{iTarget};
        
        for iSource = 1:nRegions
            
            if iSource ~= iTarget
                continue
            end
            
            if ~ismember(iTarget, find(badRgn))
                a = td.(['Curr', rgns{iTarget,1}, '_', rgns{iTarget,1}]);
                a = a(1 : newTrlInds(2) - 1, :); % get sort from first trial
                a = a./repmat(mean(abs(a), 1), size(a, 1), 1);
                idx = 1 : size(a, 2);
            else
                continue
            end
            
            in_source = inArrays{iSource};
            
            if sum(in_target) >= n_PCs && sum(in_source) >= n_PCs
                
                subplot(AxCurr(iSource));
                
                % divide by mean(abs(val)) of current for each unit
                P = td.(['Curr', rgns{iSource,1}, '_', rgns{iTarget,1}]);
                P = P(1 : newTrlInds(nD + 2) - 1, :); % do first nD + 1 trls
                P = P(:,idx); P = P ./ mean(abs(P),1);
                
                allP = [allP; P(:)];
                
                imagesc(P'); axis tight
                
                for s = 2 : nD + 1
                    line(gca, [newTrlInds(s), newTrlInds(s)], get(gca, 'ylim'), 'color', 'black', 'linewidth', 1.25)
                end
                
                title([rgns{iSource,1} ' > ' rgns{iTarget,1}],'FontWeight', 'bold')
                xlabel('trials'), ylabel('neurons')
                
                if count==2
                    text(AxCurr(count), -(1.25) * mean(get(AxCurr(count),'xlim')), -25, ...
                        [allFiles{f}, ': intra-rgn currents over 1st ', num2str(nD + 1), ' trls'], 'fontweight', 'bold', 'fontsize', 13)
                end
                
                if iSource == 2
                    [oldxlim, oldylim] = size(P);
                end
                
            end
            
            count = count + 1;
            
        end
        
        
    end
    
    % axis(AxCurr, 'tight'),
    % colormap(cm);
    % set(gcf, 'units', 'normalized', 'outerposition', [0.05 0.1 1 0.9])
    % c = [-3.5 3.5];
    
    [~, L, U, C] = isoutlier(allP, 'percentile', [0.5 99.5]);
    c = [-1 * max(abs([L, U])), max(abs([L, U]))];
    set(AxCurr, 'clim', c)
    oldpos = get(AxCurr(2),'Position');
    colorbar(AxCurr(2)), colormap(cm);
    set(AxCurr(2),'Position', oldpos)
    set(AxCurr(2), 'xlim', [0 oldxlim], 'ylim', [0 oldylim])
    
    print('-dtiff', '-r400', [RNNfigdir, allFiles{f}, '_intraRgnCurrents_consecutiveTrls'])
    
    
    %% plot norm of projection of top n_PCs (from all trials) of currents(line) - THIS IS THE PROBLEM - NORM OF THE TOP 10 PCs!
    
    %    close all
    %
    %     for iTarget = 1:nRegions % One plot per target
    %         in_target = inArrays{iTarget};
    %
    %         figure('color','w');
    %         AxNormProj = arrayfun(@(i) subplot(4,2,i,'NextPlot', 'add', 'Box', 'off',  'TickDir', 'out', 'FontSize', 10, 'FontWeight', 'bold', ...
    %             'xtick', newTrlInds(1:2:end), 'xticklabel', 1:2:nTrls), 1:nRegions);
    %         count = 1;
    %
    %         for iSource = 1:nRegions
    %             in_source = inArrays{iSource};
    %
    %             if sum(in_target) >= n_PCs && sum(in_source) >= n_PCs
    %
    %                 subplot(AxNormProj(iSource));
    %
    %                 P = td.(['Curr' rgns{iSource,1}, '_', rgns{iTarget,1}, '_pca_norm']);
    %                 P = cell2mat(P);
    %                 P = P - repmat(P(1,:),size(P,1),1);
    %                 plot(P(:, 1),'LineWidth', 2, 'Color', rgnColors(iSource,:))
    %
    %                 if min(firstTrlNewSet) <= max(newTrlInds)
    %                     for s = 1:length(firstTrlNewSet)
    %                         line(gca, [newTrlInds(firstTrlNewSet(s)), newTrlInds(firstTrlNewSet(s))], [-2 2], 'color', 'black', 'linewidth', 1.5)
    %
    %                         if s ~= length(firstTrlNewSet)
    %                             setLabelPos = 0.5 * (newTrlInds(firstTrlNewSet(s)) + newTrlInds(firstTrlNewSet(s+1)));
    %                         else
    %                             setLabelPos = 0.5 * (newTrlInds(firstTrlNewSet(s)) + length(DFull));
    %                         end
    %
    %                         text(gca, setLabelPos, -2.7, ['set ', num2str(allSetIDs(s + 1))], 'fontweight', 'bold')
    %                     end
    %                 end
    %
    %
    %                 title([rgns{iSource,1} ' > ' rgns{iTarget,1}],'FontWeight', 'bold')
    %                 xlabel('trials')
    %
    %                 if count==2
    %                     text(gca, -(1) * mean(get(gca,'xlim')), 1.5 * max(get(gca,'ylim')), ...
    %                         [allFiles{f}, ' norm of projection of top ', num2str(n_PCs), ' PCs to ', rgns{iTarget, 1}], 'fontweight', 'bold', 'fontsize', 13)
    %                 end
    %
    %                 count = count + 1;
    %
    %             end
    %         end
    %
    %         axis(AxNormProj, 'tight'),
    %         set(gcf, 'units', 'normalized', 'outerposition', [0.05 0.1 1 0.9])
    %
    %         ymin = round(min(min(cell2mat(get(AxNormProj,'ylim')))),2,'decimals');
    %         ymax = round(max(max(cell2mat(get(AxNormProj,'ylim')))),2,'decimals');
    %         % set(AxNormProj,'ylim', [ymin ymax])
    %         set(AxNormProj,'ylim', [-2 2])
    %         arrayfun(@(s) line(AxNormProj(s), get(AxNormProj(s),'xlim'), [0 0], 'linestyle', ':', 'color','black','linewidth', 1.5), 1:length(AxNormProj))
    %
    %         print('-dtiff', '-r400', [RNNfigdir, allFiles{f}, '_pca_normed_currents_to_', rgns{iTarget,1},])
    %         close
    %     end
    %
    
    
    %% consecutive trials from first set
    
    cm = brewermap(100, '*RdBu');
    
    
    figT = figure('color','w');
    AxT = arrayfun(@(i) subplot(sum(rgnIxToPlot), nD, i, 'NextPlot', 'add', 'Box', 'on', 'BoxStyle', 'full', 'linewidth', 1, ...
        'xtick', '', 'xticklabel', '', ...
        'ytick', '', 'ydir', 'reverse'), 1:((sum(rgnIxToPlot) * nD)));
    set(gcf, 'units', 'normalized', 'outerposition', [0 0.2 1 0.6])
    count = 1;
    dJ_consecutive = [];
    colormap(cm)
    for iRgn = find(rgnIxToPlot) + 1
        if ~ismember(iRgn - 1, find(rgnIxToPlot))
            continue
        end
        
        in_rgn = newRgnInds(iRgn - 1) + 1 : newRgnInds(iRgn);
        
        for i = 1 : nD
            
            J_prev = squeeze(Jordr(in_rgn, in_rgn, trlsToPlot(i))); % trained J from first trial of current set
            J_curr = squeeze(Jordr(in_rgn, in_rgn, trlsToPlot(i) + 1)); % trained J from second trial of current set
            nUnitsRgn = numel(in_rgn);
            assert(isequal(nUnitsRgn, nUnitsAll(iRgn - 1)))
            J_delta = J_curr - J_prev;
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
                presynLbl = text(AxT(count), round(nUnitsRgn/3), -10, 'pre-syn', 'fontsize', 10, 'fontweight', 'bold');
                postsynLbl = text(AxT(count), round(1.06*nUnitsRgn), round(nUnitsRgn/3), 'post-syn', 'fontsize', 10, 'fontweight', 'bold');
                set(postsynLbl, 'rotation', 270, 'horizontalalignment', 'left')
            end
            
            if i == 1
                ylabel([rgnLabels{iRgn-1}, ' (', num2str(nUnitsRgn), ' units)'], 'fontweight', 'bold', 'fontsize', 13)
                set(get(AxT(count), 'ylabel'), 'rotation', 90, 'horizontalalignment', 'center')
            end
            
            if count >= length(AxT) - (nD) + 1
                xlabel([num2str(trlsToPlot(i) + 1), '\Delta', num2str(trlsToPlot(i))], 'fontsize', 12, 'fontweight', 'bold')
            end
            
            count = count + 1;
            
        end
        
    end
    
    % update clims for the last three
    [~, L, U, C] = isoutlier(dJ_consecutive, 'percentile', [0.5 99.5]);
    newCLims =  [-0.46 0.46];
    % newCLims = [-1 * max(abs([L, U])), max(abs([L, U]))];
    set(AxT, 'clim', newCLims),
    titleAxNum = round(0.5*(nD));
    text(AxT(titleAxNum ), -25, -15, ...
        [allFiles{f},': within set \Delta Js, consecutive (range \Delta +/-', num2str(newCLims(2), '%.2f'), ')'], 'fontweight', 'bold', 'fontsize', 14)
    
    set(figT, 'currentaxes', AxT),
    print(figT, '-dtiff', '-r400', [RNNfigdir, allFiles{f}, '_intraRgnDeltaJs_consecutiveTrls']),
    close(figT)
    
    
    
    %% BETWEEN TRIAL INTRA REGIONAL UPDATES
    
%     % 1 of 3 - intra-rgn delta Js from end prev set and start new set
%     cm = brewermap(100, '*RdBu');
%     figD = figure('color','w');
%     AxD = arrayfun(@(i) subplot(nRegions - sum(badRgn), nDeltasToPlot, i, 'NextPlot', 'add', 'Box', 'off', ...
%         'xtick', '', 'xticklabel', '', ...
%         'ytick', '', 'ydir', 'reverse'), 1:((1 * nRegions - sum(badRgn)) * (nDeltasToPlot)));
%     set(gcf, 'units', 'normalized', 'outerposition', [0 0 1 1])
%     count = 1;
%     indsSetsToPlot = round(linspace(1, length(firstTrlNewSet), nDeltasToPlot));
%     trlsToPlot = firstTrlNewSet(indsSetsToPlot);
%     allJ_bw_delta = [];
%     for iRgn = 2 : nRegions
%         
%         in_rgn = newRgnInds(iRgn - 1) + 1 : newRgnInds(iRgn);
%         
%         for i = 1 : nDeltasToPlot
%             
%             J_prev = squeeze(Jordr(in_rgn, in_rgn, trlsToPlot(i) - 1)); % trained J from last trial of previous set
%             J_curr = squeeze(Jordr(in_rgn, in_rgn, trlsToPlot(i))); % trained J from first trial of current set
%             nUnitsRgn = numel(in_rgn);
%             assert(isequal(nUnitsRgn, nUnitsAll(iRgn - 1)))
%             J_delta = J_curr - J_prev;
%             allJ_bw_delta = [allJ_bw_delta; J_delta(:)];
%             subplot(AxD(count)), imagesc(J_delta); axis tight
%             
%             if count == 1
%                 presynLbl = text(AxD(count), round(nUnitsRgn/3), -5, 'pre-syn', 'fontsize', 10);
%                 postsynLbl = text(AxD(count), round(1.05*nUnitsRgn), round(nUnitsRgn/3), 'post-syn', 'fontsize', 10);
%                 set(postsynLbl, 'rotation', 270, 'horizontalalignment', 'left')
%             end
%             
%             if i == 1
%                 ylabel(rgnLabels(iRgn-1), 'fontweight', 'bold', 'fontsize', 14)
%                 set(get(AxD(count), 'ylabel'), 'rotation', 0, 'horizontalalignment', 'right')
%             end
%             
%             if count >= length(AxD) - (nDeltasToPlot) + 1
%                 xlabel(['\DeltaJ (trl ', num2str(trlsToPlot(i)), 'vs', num2str(trlsToPlot(i) - 1), ')'], 'fontweight', 'bold', 'fontsize', 12)
%             end
%             
%             if mod(count,nDeltasToPlot) == 0 && iRgn == 2
%                 oldpos = get(AxD(count),'Position');
%                 cb = colorbar(AxD(count));
%                 colormap(cm)
%                 cbpos = get(cb, 'position');
%                 set(cb, 'position', [1.025 * (oldpos(1) + oldpos(3)), cbpos(2:end)])
%                 set(AxD(count),'Position', oldpos)
%             end
%             
%             count = count + 1;
%             
%         end
%         
%     end
%     
%     titleAxNum = round(0.5*(nDeltasToPlot));
%     text(AxD(titleAxNum ), -25, -20, ...
%         [allFiles{f},': between set \Deltatrained Js (curr set VS prev set)'], 'fontweight', 'bold', 'fontsize', 14)
%     
%     
%     % 2 of 3 - intra-rgn delta Js from first update (first and second trials) in a set
%     
%     cm = brewermap(100, '*RdBu');
%     figF =figure('color','w');
%     AxF = arrayfun(@(i) subplot(nRegions - sum(badRgn), nDeltasToPlot, i, 'NextPlot', 'add', 'Box', 'off', ...
%         'xtick', '', 'xticklabel', '', ...
%         'ytick', '', 'ydir', 'reverse'), 1:((1 * nRegions - sum(badRgn)) * (nDeltasToPlot)));
%     set(gcf, 'units', 'normalized', 'outerposition', [0 0 1 1])
%     count = 1;
%     indsSetsToPlot = round(linspace(1, length(firstTrlNewSet), nDeltasToPlot));
%     trlsToPlot = firstTrlNewSet(indsSetsToPlot);
%     allJ_first_delta = [];
%     for iRgn = 2 : nRegions
%         
%         in_rgn = newRgnInds(iRgn - 1) + 1 : newRgnInds(iRgn);
%         
%         for i = 1 : nDeltasToPlot
%             
%             J_prev = squeeze(Jordr(in_rgn, in_rgn, trlsToPlot(i))); % trained J from first trial of current set
%             J_curr = squeeze(Jordr(in_rgn, in_rgn, trlsToPlot(i) + 1)); % trained J from second trial of current set
%             nUnitsRgn = numel(in_rgn);
%             assert(isequal(nUnitsRgn, nUnitsAll(iRgn - 1)))
%             J_delta = J_curr - J_prev;
%             allJ_first_delta = [allJ_first_delta; J_delta(:)];
%             subplot(AxF(count)), imagesc(J_delta); axis tight
%             
%             if count == 1
%                 presynLbl = text(AxF(count), round(nUnitsRgn/3), -5, 'pre-syn', 'fontsize', 10);
%                 postsynLbl = text(AxF(count), round(1.05*nUnitsRgn), round(nUnitsRgn/3), 'post-syn', 'fontsize', 10);
%                 set(postsynLbl, 'rotation', 270, 'horizontalalignment', 'left')
%             end
%             
%             if i == 1
%                 ylabel(rgnLabels(iRgn-1), 'fontweight', 'bold', 'fontsize', 14)
%                 set(get(AxF(count), 'ylabel'), 'rotation', 0, 'horizontalalignment', 'right')
%             end
%             
%             if count >= length(AxF) - (nDeltasToPlot) + 1
%                 xlabel(['\DeltaJ (trl ', num2str(trlsToPlot(i) + 1), 'vs', num2str(trlsToPlot(i)), ')'], 'fontweight', 'bold', 'fontsize', 12)
%             end
%             
%             if mod(count,nDeltasToPlot) == 0 && iRgn == 2
%                 oldpos = get(AxF(count),'Position');
%                 cb = colorbar(AxF(count));
%                 colormap(cm)
%                 cbpos = get(cb, 'position');
%                 set(cb, 'position', [1.025 * (oldpos(1) + oldpos(3)), cbpos(2:end)])
%                 set(AxF(count),'Position', oldpos)
%             end
%             
%             count = count + 1;
%             
%         end
%         
%     end
%     
%     titleAxNum = round(0.5*(nDeltasToPlot));
%     text(AxF(titleAxNum ), -25, -20, ...
%         [allFiles{f},': within set \Deltatrained Js (first update)'], 'fontweight', 'bold', 'fontsize', 14)
%     
%     
%     % 3 of 3 - intra-rgn delta Js from last update (last and next to last trls) in a set
%     
%     cm = brewermap(100, '*RdBu');
%     figL = figure('color','w');
%     AxL = arrayfun(@(i) subplot(nRegions - sum(badRgn), nDeltasToPlot, i, 'NextPlot', 'add', 'Box', 'off', ...
%         'xtick', '', 'xticklabel', '', ...
%         'ytick', '', 'ydir', 'reverse'), 1:((1 * nRegions - sum(badRgn)) * (nDeltasToPlot)));
%     set(gcf, 'units', 'normalized', 'outerposition', [0 0 1 1])
%     count = 1;
%     indsSetsToPlot = round(linspace(1, length(firstTrlNewSet), nDeltasToPlot));
%     trlsToPlot = lastTrlEachSet(indsSetsToPlot);
%     allJ_last_delta = [];
%     for iRgn = 2 : nRegions
%         
%         in_rgn = newRgnInds(iRgn - 1) + 1 : newRgnInds(iRgn);
%         
%         for i = 1 : nDeltasToPlot
%             
%             J_prev = squeeze(Jordr(in_rgn, in_rgn, trlsToPlot(i) - 1)); % trained J from next to last trial of currernt set
%             J_curr = squeeze(Jordr(in_rgn, in_rgn, trlsToPlot(i))); % trained J from last trial of current set
%             nUnitsRgn = numel(in_rgn);
%             assert(isequal(nUnitsRgn, nUnitsAll(iRgn - 1)))
%             J_delta = J_curr - J_prev;
%             allJ_last_delta = [allJ_last_delta; J_delta(:)];
%             subplot(AxL(count)), imagesc(J_delta); axis tight
%             
%             if count == 1
%                 presynLbl = text(AxL(count), round(nUnitsRgn/3), -5, 'pre-syn', 'fontsize', 10);
%                 postsynLbl = text(AxL(count), round(1.05*nUnitsRgn), round(nUnitsRgn/3), 'post-syn', 'fontsize', 10);
%                 set(postsynLbl, 'rotation', 270, 'horizontalalignment', 'left')
%             end
%             
%             if i == 1
%                 ylabel(rgnLabels(iRgn-1), 'fontweight', 'bold', 'fontsize', 14)
%                 set(get(AxL(count), 'ylabel'), 'rotation', 0, 'horizontalalignment', 'right')
%             end
%             
%             if count >= length(AxL) - (nDeltasToPlot) + 1
%                 xlabel(['\DeltaJ (trl ', num2str(trlsToPlot(i)), 'vs', num2str(trlsToPlot(i) - 1), ')'], 'fontweight', 'bold', 'fontsize', 12)
%             end
%             
%             if mod(count,nDeltasToPlot) == 0 && iRgn == 2
%                 oldpos = get(AxL(count),'Position');
%                 cb = colorbar(AxL(count));
%                 colormap(cm)
%                 cbpos = get(cb, 'position');
%                 set(cb, 'position', [1.025 * (oldpos(1) + oldpos(3)), cbpos(2:end)])
%                 set(AxL(count),'Position', oldpos)
%             end
%             
%             count = count + 1;
%             
%         end
%         
%     end
%     
%     titleAxNum = round(0.5*(nDeltasToPlot));
%     text(AxL(titleAxNum ), -25, -20, ...
%         [allFiles{f},': within set \Deltatrained Js (last update)'], 'fontweight', 'bold', 'fontsize', 14)
%     
%     % update clims for the last three
%     X = [allJ_bw_delta; allJ_first_delta; allJ_last_delta];
%     [~, L, U, C] = isoutlier(X, 'percentile', [0.5 99.5]);
%     newCLims = [-1 * max(abs([L, U])), max(abs([L, U]))];
%     
%     set(figD, 'currentaxes', AxD),
%     set(AxD, 'clim', newCLims),
%     print(figD, '-dtiff', '-r400', [RNNfigdir, allFiles{f}, '_intraRgnDeltaJs_betweenSets']),
%     close(figD)
%     
%     set(figF, 'currentaxes', AxF),
%     set(AxF, 'clim', newCLims),
%     print(figF, '-dtiff', '-r400', [RNNfigdir, allFiles{f}, '_intraRgnDeltaJs_firstUpdateInSet'])
%     close(figF)
%     
%     set(figL, 'currentaxes', AxL),
%     set(AxL, 'clim', newCLims),
%     print(figL, '-dtiff', '-r400', [RNNfigdir, allFiles{f}, '_intraRgnDeltaJs_lastUpdateInSet']), close(figL)
%     why
%     
%     
%     %% imagesc delta Js end of a set to start of a set
%     
%     cm = brewermap(100, '*RdBu');
%     
%     for iTarget = 1 : nRegions
%         
%         in_target = inArrays{iTarget};
%         nTarg = sum(in_target);
%         
%         figure('color','w');
%         AxJ = arrayfun(@(i) subplot(nRegions - sum(badRgn), nDeltasToPlot, i, 'NextPlot', 'add', 'Box', 'off', ...
%             'xtick', '', 'ytick', '', 'ydir', 'reverse'), 1:((1 * nRegions - sum(badRgn)) * (nDeltasToPlot)));
%         set(gcf, 'units', 'normalized', 'outerposition', [0 0 1 1])
%         
%         count = 1;
%         setsToPlot = 1 : xthSet : nSets - 1;
%         
%         for iSource = 1 : nRegions
%             
%             in_source = inArrays{iSource};
%             nSource = sum(in_source);
%             
%             if nTarg >= n_PCs && nSource >= n_PCs
%                 
%                 for i = 1 : nDeltasToPlot
%                     
%                     J_curr = squeeze(J(in_target, in_source, firstTrlNewSet(setsToPlot(i))));
%                     J_next = squeeze(J(in_target, in_source, lastTrlEachSet(setsToPlot(i))));
%                     subplot(AxJ(count)), imagesc(J_next - J_curr); axis tight;
%                     presynLbl = text(AxJ(count), round(nSource/3), -5, 'pre-syn', 'fontsize', 10);
%                     postsynLbl = text(AxJ(count), round(1.05*nSource), round(nTarg/3), 'post-syn', 'fontsize', 10);
%                     set(postsynLbl, 'rotation', 270, 'horizontalalignment', 'left')
%                     
%                     if i == 1
%                         ylabel([rgns{iSource,1}, ' > '], 'fontweight', 'bold')
%                         set(get(gca, 'ylabel'), 'rotation', 0, 'horizontalalignment', 'right')
%                     end
%                     
%                     if count >= length(AxJ) - (nDeltasToPlot) + 1
%                         xlabel(['finalJ\DeltastartJ (set ', num2str(setsToPlot(i)), ')'], 'fontweight', 'bold', 'fontsize', 12)
%                     end
%                     
%                     if mod(count,nDeltasToPlot) == 0
%                         oldpos = get(AxJ(count),'Position');
%                         cb = colorbar(AxJ(count));
%                         colormap(cm)
%                         cbpos = get(cb, 'position');
%                         set(cb, 'position', [1.025 * (oldpos(1) + oldpos(3)), cbpos(2:end)])
%                         set(AxJ(count),'Position', oldpos)
%                     end
%                     
%                     count = count + 1;
%                     
%                 end
%             end
%             
%         end
%         
%         % set each row to have different clims (by source)
%         % arrayfun(@(i) set(AxJ(i:i+nDeltasToPlot-1), 'clim', ...
%         % [max(max(abs(cell2mat(get(AxJ(i:i+nDeltasToPlot-1), 'clim'))))), max(max(abs(cell2mat(get(AxJ(i:i+nDeltasToPlot-1), 'clim')))))]), 1:nDeltasToPlot:length(AxJ));
%         set(AxJ, 'clim', [-1 1])
%         
%         titleAxNum = round(0.5*(nDeltasToPlot));
%         text(AxJ(titleAxNum ), -15, -20, ...
%             [allFiles{f},': start set to end set \Deltafinal Js to rgn -> ',  rgns{iTarget,1}], 'fontweight', 'bold', 'fontsize', 14)
%         print('-dtiff', '-r400', [RNNfigdir, allFiles{f}, '_deltaJs_endSetStartSet_to_', rgns{iTarget,1},])
%         close
%         
%     end
%     
%     %% imagesc delta Js end of a trial vs beginning of that trl
%     
%     cm = brewermap(100, '*RdBu');
%     
%     for iTarget = 1 : nRegions
%         
%         in_target = inArrays{iTarget};
%         nTarg = sum(in_target);
%         
%         figure('color','w');
%         AxJ = arrayfun(@(i) subplot(nRegions - sum(badRgn), nDeltasToPlot, i, 'NextPlot', 'add', 'Box', 'off', ...
%             'xtick', '', 'ytick', '', 'ydir', 'reverse'), 1:((1 * nRegions - sum(badRgn)) * (nDeltasToPlot)));
%         set(gcf, 'units', 'normalized', 'outerposition', [0 0 1 1])
%         
%         count = 1;
%         trlsToPlot = 1 : xthTrl : nTrls;
%         
%         for iSource = 1 : nRegions
%             
%             in_source = inArrays{iSource};
%             nSource = sum(in_source);
%             
%             if nTarg >= n_PCs && nSource >= n_PCs
%                 
%                 for i = 1 : nDeltasToPlot
%                     
%                     J_beg = squeeze(J0(in_target, in_source, trlsToPlot(i)));
%                     J_end = squeeze(J(in_target, in_source, trlsToPlot(i)));
%                     subplot(AxJ(count)), imagesc(J_end - J_beg); axis tight;
%                     presynLbl = text(AxJ(count), round(nSource/3), -5, 'pre-syn', 'fontsize', 10);
%                     postsynLbl = text(AxJ(count), round(1.05*nSource), round(nTarg/3), 'post-syn', 'fontsize', 10);
%                     set(postsynLbl, 'rotation', 270, 'horizontalalignment', 'left')
%                     
%                     if i == 1
%                         ylabel([rgns{iSource,1}, ' > '], 'fontweight', 'bold')
%                         set(get(gca, 'ylabel'), 'rotation', 0, 'horizontalalignment', 'right')
%                     end
%                     
%                     if count >= length(AxJ) - (nDeltasToPlot) + 1
%                         xlabel(['J\DeltaJ0 (trl ', num2str(trlsToPlot(i)),')'], 'fontweight', 'bold', 'fontsize', 12)
%                     end
%                     
%                     if mod(count,nDeltasToPlot) == 0
%                         oldpos = get(AxJ(count),'Position');
%                         cb = colorbar(AxJ(count));
%                         cbpos = get(cb, 'position');
%                         set(cb, 'position', [1.025 * (oldpos(1) + oldpos(3)), cbpos(2:end)])
%                         colormap(cm)
%                         set(AxJ(count),'Position', oldpos)
%                     end
%                     
%                     count = count + 1;
%                     
%                 end
%             end
%             
%         end
%         
%         % set each row to have different clims (by source)
%         % arrayfun(@(i) set(AxJ(i:i+nDeltasToPlot-1), 'clim', ...
%         % [-1*max(max(abs(cell2mat(get(AxJ(i:i+nDeltasToPlot-1), 'clim'))))), max(max(abs(cell2mat(get(AxJ(i:i+nDeltasToPlot-1), 'clim')))))]), 1:nDeltasToPlot:length(AxJ));
%         
%         % arrayfun(@(i) set(AxJ(i:i+nDeltasToPlot-1), 'clim', ...
%         % [min(min(cell2mat(get(AxJ(i:i+nDeltasToPlot-1), 'clim')))), max(max(cell2mat(get(AxJ(i:i+nDeltasToPlot-1), 'clim'))))]), 1:nDeltasToPlot:length(AxJ));
%         set(AxJ, 'clim', [-1 1])
%         
%         titleAxNum = round(0.5*(nDeltasToPlot));
%         text(AxJ(titleAxNum ), -15, -20, ...
%             [allFiles{f},': within trial \DeltaJs to rgn ',  rgns{iTarget,1}], 'fontweight', 'bold', 'fontsize', 14)
%         print('-dtiff', '-r400', [RNNfigdir, allFiles{f}, '_JdeltaJ0s_endTrlStartTrl_to_', rgns{iTarget,1},])
%         close
%         
%     end
%     
%     %% boxplots - final J of last trial each set, full J
%     
%     cm = colormap(cool);
%     nUnits = size(J,1);
%     
%     if exist('colororder','file') ~= 0
%         histGradColors3 = cm(round(linspace(1, size(cm, 1), nSets - 1)),:);
%     end
%     
%     J_allsets = NaN(nSets - 1, nUnits * nUnits);
%     
%     for i = 1 : nSets - 1
%         J_curr = squeeze(J(:, :, lastTrlEachSet(i)));
%         J_allsets(i, :) = reshape(J_curr(:) ./ sqrt(nUnits), nUnits * nUnits, 1);
%         
%     end
%     
%     J_resh = reshape(J_allsets', 1, (nSets - 1) * (nUnits * nUnits));
%     J_setLbl = repelem(allSetIDs(2:end), nUnits * nUnits);
%     boxTbl = table(J_resh, J_setLbl, 'VariableNames', {'J_vals', 'set_ID'});
%     figure('color','w');
%     set(gcf, 'units', 'normalized', 'outerposition', [0.02 0.1 0.95 0.8])
%     
%     axB = axes('NextPlot','add','FontSize',14, 'fontweight', 'bold', 'TickDir','out');
%     boxplot( boxTbl.J_vals, boxTbl.set_ID, 'Whisker', 5, 'Parent', axB );
%     boxLines = findobj(axB,'Type','Line');
%     arrayfun( @(x) set(x,'LineStyle','-','Color','k','LineWidth',1), boxLines )
%     boxObjs = findobj(axB,'Tag','Box');
%     arrayfun( @(iBox) patch( boxObjs(iBox).XData, boxObjs(iBox).YData, histGradColors3(iBox, :), 'FaceAlpha', 0.5), 1 : nSets - 1 )
%     grid minor
%     
%     
%     colormap cool
%     scatter(boxTbl.set_ID, boxTbl.J_vals, 30, linspace(1, nSets - 1, (nSets - 1) * nUnits * nUnits), 'filled', ...
%         'jitter', 'on', 'jitterAmount', 0.4, ...
%         'markerfacealpha', 0.5)
%     grid minor
%     set(axB, 'xlim', [0.5 nSets-0.5])
%     xlabel('# set'),
%     ylabel('interaction strength')
%     title([allFiles{f}, ': J distributions at end of set'])
%     
%     
%     print('-dtiff', '-r400', [RNNfigdir, allFiles{f}, '_boxplots_endOfSetJOverSets'])
%     close
%     
    %%
    %     dJPlotLabels = [cellstr(repmat('b/w set update', size(allJ_bw_delta))); ...
    %         cellstr(repmat('first update in set', size(allJ_first_delta))); ...
    %         cellstr(repmat('last update in set', size(allJ_last_delta)))];
    %     dJXPos = [ones(size(allJ_bw_delta)); repmat(2, size(allJ_first_delta)); repmat(3, size(allJ_last_delta))];
    %
    %     dJ = [allJ_bw_delta; allJ_first_delta; allJ_last_delta];
    %     boxTbl2 = table(dJ, dJPlotLabels, 'VariableNames', {'delta_Js','version'});
    %
    %     figure('color','w');
    %     set(gcf, 'units', 'normalized', 'outerposition', [0.02 0.1 0.95 0.8])
    %
    %     axB2 = axes('NextPlot','add','FontSize',14, 'fontweight', 'bold', 'TickDir','out');
    %     boxplot(boxTbl2.delta_Js, boxTbl2.version, 'Whisker', 5, 'Parent', axB2)
    %
    %     boxLines = findobj(axB2,'Type','Line');
    %     arrayfun( @(x) set(x,'LineStyle','-','Color','k','LineWidth',1), boxLines )
    %     boxObjs = findobj(axB2,'Tag','Box');
    %     arrayfun( @(iBox) patch( boxObjs(iBox).XData, boxObjs(iBox).YData, histGradColors3(iBox, :), 'FaceAlpha', 0.5), 1 : 3 )
    %     grid minor
    %
    %     colormap cool
    %     scatter(dJXPos, boxTbl2.delta_Js, 'filled', ...
    %         'jitter', 'on', 'jitterAmount', 0.4, ...
    %         'markerfacealpha', 0.5)
    %     grid minor
    %     set(axB, 'xlim', [0.5 nSets-0.5])
    %     ylabel('\DeltaJ')
    
    
    
end

