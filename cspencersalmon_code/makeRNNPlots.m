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
    
    [~, L, U, C] = isoutlier(allP, 'percentile', [0.5 99.5]);
    c = [-1 * max(abs([L, U])), max(abs([L, U]))];
    set(AxCurr, 'clim', c)
    oldpos = get(AxCurr(2),'Position');
    colorbar(AxCurr(2)), colormap(cm);
    set(AxCurr(2),'Position', oldpos)
    set(AxCurr(2), 'xlim', [0 oldxlim], 'ylim', [0 oldylim])
    
    print('-dtiff', '-r400', [RNNfigdir, allFiles{f}, '_intraRgnCurrents_consecutiveTrls'])
    
    
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
    
    
end

