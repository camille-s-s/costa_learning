clearvars;
close all

% params
aFac            = 0.01; % idk comes from softNormalize > o3_AddCurrentsToTD_runs
center_data     = true;
n_PCs           = 10;
doSmooth        = true;
rgnOrder        = [1 8 2 7 4 5 6 3]; % for 4 row 2 col plots
nDeltasToPlot   = 10;

% in and outdirs
bd              = '~/Dropbox (BrAINY Crew)/costa_learning/';
mdlDir          = [bd 'models/'];
RNNfigdir       = [bd 'models/figures/'];
spikeInfoPath   = [bd 'reformatted_data/'];

%%
addpath(genpath(bd))
cd(mdlDir)
mdlFiles = dir('rnn_*_set*_trial*.mat');

allFiles = unique(arrayfun(@(i) ...
    mdlFiles(i).name(strfind(mdlFiles(i).name, 'rnn') + 4 : strfind(mdlFiles(i).name, 'set') - 2), 1:length(mdlFiles), 'un', false));

for f = 1 : length(allFiles) % for each session....
    
    currSsn = dir(['rnn_', allFiles{f}, '_*.mat']);
    allSetIDs = unique(arrayfun(@(i) ...
        str2double(currSsn(i).name(strfind(currSsn(i).name,'set') + 3 : strfind(currSsn(i).name,'trial') - 2)), 1:length(currSsn)));
    allTrialIDs = unique(arrayfun(@(i) ...
        str2double(currSsn(i).name(strfind(currSsn(i).name,'trial') + 5 : end - 4)), 1:length(currSsn)));
    
    nTrls = length(allTrialIDs);
    xthTrl = round((nTrls - 1) / nDeltasToPlot);
    nSets = length(allSetIDs);
    % nTrls = 1;
    
    J = [];
    J0 = [];
    D = cell(nTrls, 1);
    R_ds = cell(nTrls, 1);
    tData = cell(nTrls, 1);
    setID = [];
    trlNum = [];
    
    for i = 1 : nTrls % for each trial in a session
        
        mdlfnm = [mdlDir, currSsn(i).name];
        load(mdlfnm)
        
        mdl             = RNN.mdl;
        J(:, :, i)      = mdl.J;
        J0(:, :, i)     = mdl.J0;
        D{i}            = mdl.targets;
        setID(i)        = mdl.setID;
        trlNum(i)       = mdl.iTrl;
        
        % get indices for each sample of model data
        tData{i}        = mdl.tData;
        R_ds{i}         = mdl.RMdlSample;
        
        if i == 1 % assume or check if same for all
            
            spikeInfoName = [currSsn(i).name(5 : median(strfind(currSsn(i).name, '_'))-1), '_meta.mat'];
            load([spikeInfoPath, spikeInfoName], 'spikeInfo')
            binSize = mdl.dtData; % in sec
            
            % set up indexing vectors for submatrices
            rgns            = mdl.params.arrayRegions;
            dtData          = mdl.dtData;
            arrayList       = rgns(:, 2);
            nRegions        = length(arrayList);
            rgnColors       = brewermap(nRegions, 'Spectral');% cmap(round(linspace(1, 255, nRegions)),:);
        end
    end
    
    % reorder by trlNum
    [~, trlSort] = sort(trlNum, 'ascend');
    J = J(:, :, trlSort);
    J0 = J0(:, :, trlSort);
    D = D(trlSort);
    setID = setID(trlSort);
    tData = tData(trlSort);
    R_ds = R_ds(trlSort);
    
    % make regions more legible
    rgns(:,1) = arrayfun(@(iRgn) strrep(rgns{iRgn, 1}, 'left_', 'L'), 1 : nRegions, 'un', false)';
    rgns(:,1) = arrayfun(@(iRgn) strrep(rgns{iRgn, 1}, 'right_', 'R'), 1 : nRegions, 'un', false)';
    rgns = rgns(rgnOrder, :);
    
    DFull = cell2mat(D');
    nSPFull = size(DFull,2);
    nSPTmp = cell2mat(cellfun(@size, D, 'un', 0));
    nSP = nSPTmp(:, 2);
    newTrlInds = [1; cumsum(nSP(1:end-1))+1]; % for demarcating new trials in later plotting
    trlStartsTmp = [newTrlInds - 1; size(DFull, 2)];
    trlStarts = cell2mat(arrayfun(@(iTrl) [trlStartsTmp(iTrl, 1)+1 trlStartsTmp(iTrl+1, 1)], 1:nTrls, 'un', 0)');
    
    % for trial averaging
    shortestTrl = min(diff(newTrlInds)) - 1;
    DTrunc = D;
    DTrunc = arrayfun(@(iTrl) DTrunc{iTrl}(:, 1 : shortestTrl), 1 : nTrls, 'un', false)';
    DTrunc = cell2mat(DTrunc');
    DMean = cell2mat(arrayfun(@(n) mean(reshape(DTrunc(n, :), shortestTrl, size(DTrunc, 2)/shortestTrl), 2)', 1 : size(DTrunc, 1), 'un', false)');
    
    % for demarcating sets
    firstTrlNewSet = cumsum(arrayfun(@(iSet) sum(setID == allSetIDs(iSet)), 1:length(allSetIDs)));
    firstTrlNewSet = firstTrlNewSet(1 : end - 1) + 1;
    
    % get the largest possible minimum number of trials from each set
    [C, ia, ic] = unique(setID');
    a_counts = accumarray(ic, 1);
    minTrlsPerSet = min(a_counts(2:end));
    
    %% semi-convert to matt format
    
    td = struct;
    
    % get spikes for all trials included, stuck end to end, and prune bad units by cutting any unit whose overall FR is below threshold
    for iRgn = 1 : nRegions
        in_rgn = rgns{iRgn, 3};
        td.([rgns{iRgn,1}, '_spikes']) = DFull(in_rgn, :)';
        td.([rgns{iRgn,1}, '_spikes_avg']) = DMean(in_rgn, :)';
    end
    
    inArrays = rgns(:, 3);
    
    % rgn w really low # units gets excluded
    badRgn = arrayfun(@(iRgn) sum(inArrays{iRgn}) < n_PCs, 1:nRegions);
    
    for iTarget = 1:nRegions
        in_target = inArrays{iTarget};
        
        for iSource = 1:nRegions
            in_source = inArrays{iSource};
            
            if sum(in_target) >= n_PCs && sum(in_source) >= n_PCs
                
                P_inhAll = cell(nTrls, 1);
                P_excAll = cell(nTrls, 1);
                P_bothAll = cell(nTrls, 1);
                
                % get currents (normalized as in softNormalize as far as I can tell...)
                for i = 1 : nTrls
                    
                    % For each trial, collect currents and combine together for PCA
                    JTrl = squeeze(J(:, :, i));
                    RTrl = R_ds{i};
                    
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
                [w, scores, eigen,~,~,mu] = pca(P_bothAll', 'Algorithm','svd', 'Centered', center_data, 'Economy',0);
                
                % for each trial, combined currents only: project into low-D space and pick the num_dims requested
                tempProjAll = cell(nTrls, 1);
                normProjAll = cell(nTrls, 1);
                
                for i = 1 : nTrls
                    
                    % project over each trial
                    projData = P_bothAll(:, trlStarts(i,1):trlStarts(i,2))'; % # sample points x neurons for a given stim
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
    
    %% pca the J space???
    
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
    
    
    %% prune bad units by cutting any unit whose overall FR is below threshold
    
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
    
    %% PCA of activity?
    d = DFull;
    bad_units = mean(d, 2) == 0;
    %d(bad_units, :) = [];
    dproj = DMean(~bad_units, :);
    
    [w, scores, eigen, ~, pvar, mu] = pca(d', 'Algorithm', 'svd', 'Centered', center_data, 'Economy', 0);
    n_PCs_act = find(cumsum(pvar)>=95,1);
    projData = dproj(:, :)';
    tempProj = (projData - repmat(mu,size(projData,1),1)) * w;
    
    % norm of n_PCs requested at each timepoint
    top_proj = tempProj(:, 1:n_PCs_act);
    normProj = zeros(size(top_proj,1),1);
    for tt = 1:size(top_proj,1)
        normProj(tt) = norm(top_proj(tt, :));
    end
    
    % project on trial average activity?
    x = tempProj;
    figure('color','w');
    cm = brewermap(size(x,1), '*RdBu');
    colormap(cm)
    patch([x(:, 1)' nan], [x(:, 2)' nan],[x(:, 3)' nan], [linspace(0, 1, size(x,1)) nan], ...
        'linewidth', 1.75, 'FaceColor', 'none', 'EdgeColor', 'interp')
    hold on
    
    patch([x(1:5:end, 1)' nan], [x(1:5:end, 2)' nan], [x(1:5:end, 3)' nan], [linspace(0, 1, numel(1:5:size(x,1))) nan], ...
        'marker', 'o', 'markersize', 6, 'markerfacecolor', 'flat', 'edgecolor', 'none')
    patch(x(1, 1), x(1, 2), x(1, 3), 0, 'marker', '^', 'markersize', 15, 'markerfacecolor', cm(1, :), 'markeredgecolor', cm(1, :))
    patch(x(end, 1), x(end, 2), x(end, 3), 1, 'marker', 's', 'markersize', 15, 'markerfacecolor', cm(end, :), 'markeredgecolor', cm(end, :))
    colorbar, grid minor
    view(3)
    print('-dtiff', '-r400', [RNNfigdir, allFiles{f}, '_top_pcs_pjn_targ_activity'])
    close
    
    %% average target activity by region
    
    % set up figure
    figure('color','w');
    set(gcf, 'units', 'normalized', 'outerposition', [0 0 1 1])
    AxTargs = arrayfun(@(i) subplot(4,2,i,'NextPlot', 'add', 'Box', 'on', 'TickDir','out', 'FontSize', 10, 'Fontweight', 'bold',  ...
        'xtick', 1:50:shortestTrl, 'xticklabel', dtData*((1:50:shortestTrl) - 1), 'ytick', ''), 1:nRegions);
    cm = brewermap(100,'*RdGy');
    
    softFac = 0.0000;
    for iRgn =  1 : nRegions
        if ~ismember(iRgn, find(badRgn))
            rgnLabel = rgns{iRgn, 1};
            rgnLabel(strfind(rgnLabel, '_')) = ' ';
            
            a = td.([rgns{iRgn,1}, '_spikes_avg']); % T x N
            
            % get the sort
            [~, idx] = max(a, [], 1);
            [~, rgnSort] = sort(idx);
            
            normfac = mean(abs(a), 1); % 1 x N
            tmpRgn = a; %./ (normfac + softFac);
            
            subplot(AxTargs(iRgn))
            imagesc(tmpRgn(:, rgnSort)' )
            axis(AxTargs(iRgn), 'tight')
            
            title(rgnLabel, 'fontweight', 'bold')
            set(gca, 'xcolor', rgnColors(iRgn,:),'ycolor', rgnColors(iRgn,:), 'linewidth', 2)
            xlabel('time (sec)'), ylabel('neurons')
            
            if iRgn == 2
                text(gca, -0.75*mean(get(gca,'xlim')), 1.1*max(get(gca,'ylim')), ...
                    [allFiles{f}, ' trial averaged target rates'], 'fontweight', 'bold', 'fontsize', 13)
                [oldxlim, oldylim] = size(tmpRgn);
            end
        end
    end
    
    set(AxTargs, 'CLim', [0 1])
    oldpos = get(AxTargs(2),'Position');
    colorbar(AxTargs(2)), colormap(cm);
    set(AxTargs(2),'Position', oldpos)
    set(AxTargs(2), 'xlim', [0 oldxlim], 'ylim', [0 oldylim])
    print('-dtiff', '-r400', [RNNfigdir, allFiles{f}, '_trl_avged_targets_by_rgn'])
    close
    
    %% plot currents (imagesc) - to do: currents over time or currents averaged?
    close all
    cm = brewermap(100,'*RdBu');
    c = [-3.5 3.5];
    
    for iTarget = 1:nRegions % One plot per target
        in_target = inArrays{iTarget};
        
        if ~ismember(iTarget, find(badRgn))
            a = td.(['Curr', rgns{iTarget,1}, '_', rgns{iTarget,1}]);
            a = a./repmat(mean(abs(a), 1), size(a, 1), 1);
            [~,idx] = max(a, [], 1);
            [~,idx] = sort(idx);
        else
            continue
        end
        
        figure('color','w');
        AxCurr = arrayfun(@(i) subplot(4,2,i,'NextPlot', 'add', 'Box', 'off',  'TickDir', 'out', 'FontSize', 10, 'Fontweight', 'bold',...
            'xtick', newTrlInds(1:2:end), 'xticklabel', 1:2:nTrls, 'ytick', '', 'CLim', c), 1:nRegions);
        count = 1;
        
        for iSource = 1:nRegions
            in_source = inArrays{iSource};
            
            if sum(in_target) >= n_PCs && sum(in_source) >= n_PCs
                
                subplot(AxCurr(iSource));
                % divide by mean(abs(val)) of current for each unit
                P = td.(['Curr', rgns{iSource,1}, '_', rgns{iTarget,1}]);
                P = P(:,idx); P = P ./ mean(abs(P),1);
                
                imagesc(P');
                
                if min(firstTrlNewSet) <= max(newTrlInds)
                    for s = 1:length(firstTrlNewSet)
                        line(gca, [newTrlInds(firstTrlNewSet(s)), newTrlInds(firstTrlNewSet(s))], get(gca, 'ylim'), 'color', 'black', 'linewidth', 1.5)
                        if s ~= length(firstTrlNewSet)
                            setLabelPos = 0.5 * (newTrlInds(firstTrlNewSet(s)) + newTrlInds(firstTrlNewSet(s+1)));
                        else
                            setLabelPos = 0.5 * (newTrlInds(firstTrlNewSet(s)) + length(DFull));
                        end
                        text(gca, setLabelPos, -15, ['set ', num2str(allSetIDs(s + 1))], 'fontweight', 'bold')
                    end
                end
                
                title([rgns{iSource,1} ' > ' rgns{iTarget,1}],'FontWeight', 'bold')
                xlabel('trials'), ylabel('neurons')
                
                if count==2
                    text(gca, -(1/2) * mean(get(gca,'xlim')), 1.2 * max(get(gca,'ylim')), ...
                        [allFiles{f}, ' currents to ', rgns{iTarget, 1}], 'fontweight', 'bold', 'fontsize', 13)
                end
                
                if iSource == 2
                    [oldxlim, oldylim] = size(P);
                end
                
            end
            
            count = count + 1;
            
        end
        
        axis(AxCurr, 'tight'),
        colormap(cm);
        set(gcf, 'units', 'normalized', 'outerposition', [0.05 0.1 1 0.9])
        oldpos = get(AxCurr(2),'Position');
        colorbar(AxCurr(2)), colormap(cm);
        set(AxCurr(2),'Position', oldpos)
        set(AxCurr(2), 'xlim', [0 oldxlim], 'ylim', [0 oldylim])
        
        print('-dtiff', '-r400', [RNNfigdir, allFiles{f}, '_curr_to_', rgns{iTarget,1}])
        
    end
    
    
    %% plot norm of projection of top n_PCs (from all trials) of currents(line) - THIS IS THE PROBLEM - NORM OF THE TOP 10 PCs!
    close all
    
    for iTarget = 1:nRegions % One plot per target
        in_target = inArrays{iTarget};
        
        figure('color','w');
        AxNormProj = arrayfun(@(i) subplot(4,2,i,'NextPlot', 'add', 'Box', 'off',  'TickDir', 'out', 'FontSize', 10, 'FontWeight', 'bold', ...
            'xtick', newTrlInds(1:2:end), 'xticklabel', 1:2:nTrls), 1:nRegions);
        count = 1;
        
        for iSource = 1:nRegions
            in_source = inArrays{iSource};
            
            if sum(in_target) >= n_PCs && sum(in_source) >= n_PCs
                
                subplot(AxNormProj(iSource));
                
                P = td.(['Curr' rgns{iSource,1}, '_', rgns{iTarget,1}, '_pca_norm']);
                P = cell2mat(P);
                P = P - repmat(P(1,:),size(P,1),1);
                plot(P(:, 1),'LineWidth', 2, 'Color', rgnColors(iSource,:))
                
                if min(firstTrlNewSet) <= max(newTrlInds)
                    for s = 1:length(firstTrlNewSet)
                        line(gca, [newTrlInds(firstTrlNewSet(s)), newTrlInds(firstTrlNewSet(s))], [-2 2], 'color', 'black', 'linewidth', 1.5)
                        
                        if s ~= length(firstTrlNewSet)
                            setLabelPos = 0.5 * (newTrlInds(firstTrlNewSet(s)) + newTrlInds(firstTrlNewSet(s+1)));
                        else
                            setLabelPos = 0.5 * (newTrlInds(firstTrlNewSet(s)) + length(DFull));
                        end
                        
                        text(gca, setLabelPos, -2.7, ['set ', num2str(allSetIDs(s + 1))], 'fontweight', 'bold')
                    end
                end
                
                
                title([rgns{iSource,1} ' > ' rgns{iTarget,1}],'FontWeight', 'bold')
                xlabel('trials')
                
                if count==2
                    text(gca, -(1) * mean(get(gca,'xlim')), 1.5 * max(get(gca,'ylim')), ...
                        [allFiles{f}, ' norm of projection of top ', num2str(n_PCs), ' PCs to ', rgns{iTarget, 1}], 'fontweight', 'bold', 'fontsize', 13)
                end
                
                count = count + 1;
                
            end
        end
        
        axis(AxNormProj, 'tight'),
        set(gcf, 'units', 'normalized', 'outerposition', [0.05 0.1 1 0.9])
        
        ymin = round(min(min(cell2mat(get(AxNormProj,'ylim')))),2,'decimals');
        ymax = round(max(max(cell2mat(get(AxNormProj,'ylim')))),2,'decimals');
        % set(AxNormProj,'ylim', [ymin ymax])
        set(AxNormProj,'ylim', [-2 2])
        arrayfun(@(s) line(AxNormProj(s), get(AxNormProj(s),'xlim'), [0 0], 'linestyle', ':', 'color','black','linewidth', 1.5), 1:length(AxNormProj))
        
        print('-dtiff', '-r400', [RNNfigdir, allFiles{f}, '_pca_normed_currents_to_', rgns{iTarget,1},])
        close
    end
    
    
    %% HISTOGRAMS OF SUBJs over sessions. consider boxplots?
    
    cm = colormap(cool);
    histGradColors = cm(round(linspace(1, size(cm, 1), nTrls - 1)),:);
    nHistBins = 100;
    
    for iTarget = 1 : nRegions
        
        in_target = inArrays{iTarget};
        nTarg = sum(in_target);
        
        Jtarg = squeeze(J(in_target, in_target, :));
        maxabsval = round(1.02 * max(abs((Jtarg(:)./sqrt(nTarg)))), 2, 'decimals');
        
        Jlim = [-maxabsval, maxabsval]; % same JLim for each target no matter the source
        histBinWidth = (Jlim(2) - Jlim(1))/nHistBins;
        edgesnew = linspace(Jlim(1), Jlim(2), nHistBins + 1);
        histcenters = edgesnew(1:end-1) + (diff(edgesnew) ./ 2);
        
        figure('color','w');
        
        %         AxJ = arrayfun(@(i) subplot(nRegions - sum(badRgn), nDeltasToPlot, i, 'NextPlot', 'add', 'Box', 'off', ...
        %             'xlim', (3/8) * Jlim, 'xtick', linspace((3/8)*Jlim(1), (3/8)*Jlim(2), 5), 'xticklabel', '', ...
        %             'yscale', 'log', 'ylim', [0.001 1], 'ytick', [0.001 0.01 0.1 1]), 1:((1 * nRegions - sum(badRgn)) * (nDeltasToPlot)));
        
        AxJ = arrayfun(@(i) subplot(nRegions - sum(badRgn), nDeltasToPlot, i, 'NextPlot', 'add', 'Box', 'off'), 1:((1 * nRegions - sum(badRgn)) * (nDeltasToPlot)));
        set(gcf, 'units', 'normalized', 'outerposition', [0 0 1 1])
        
        % xticksCell = num2cell(linspace((3/8)*Jlim(1), (3/8)*Jlim(2), 5));
        %xticksLbl = arrayfun(@(xx) num2str(xticksCell{xx}, '%.2f'), 1:length(xticksCell), 'un', false);
        count = 1;
        
        for iSource = 1 : nRegions % [1 3 5 7] % 1:nRegions
            
            in_source = inArrays{iSource};
            nSource = sum(in_source);
            
            % for s = allSetIDs(2 : end)
            % setTrlInds = [find(setID==s,1,'first'), find(setID==s,1,'last')]; % first and last trials of a set %find(setID == s);
            % J_allsets_cts = NaN(length(setTrlInds), nHistBins);
            
            if sum(in_target) >= n_PCs && sum(in_source) >= n_PCs
                
                
                % get currents (normalized as in softNormalize as far as I can tell...)
                for i = 1 : xthTrl : nTrls - 1% length(setTrlInds)% 1 : nTrls - 1
                    
                    % For each trial in a set, collect Js....
                    % J_curr = squeeze(J(in_target, in_source, setTrlInds(i)));
                    J_curr = squeeze(J(in_target, in_source, i));
                    J_next = squeeze(J(in_target, in_source, i+1));
                    % [J_curr_cts, ~] = histcounts(reshape(J_curr(:) ./ sqrt(nSource), nSource * nTarg, 1), ...
                    %    Jlim(1) : histBinWidth : Jlim(2));
                    % J_curr_cts = J_curr_cts ./ max(J_curr_cts);
                    % J_allsets_cts(i, :) = J_curr_cts;
                    
                    
                    %                     if exist('colororder','file') ~= 0
                    %                         histGradColors2 = cm(round(linspace(1, size(cm, 1), length(setTrlInds))),:);
                    %                     end
                    %
                    %                     subplot(AxJ(count)), semilogy(histcenters,J_allsets_cts, 'o-', 'linewidth', 1, 'markersize', 2)
                    %                     colororder(AxJ(count), histGradColors2);
                    
                    subplot(AxJ(count)), imagesc(J_next - J_curr)
                    %% boxplots of first trial each set
                    
                    %
                    %                     if exist('colororder','file') ~= 0
                    %                         histGradColors3 = cm(round(linspace(1, size(cm, 1), nSets - 1)),:);
                    %                     end
                    %
                    %                     J0_allsets = NaN(nSets - 1, sum(in_target) * sum(in_source));
                    %
                    %                     for i = 1 : nSets - 1
                    %                         J0_curr = squeeze(J0(in_target, in_source, firstTrlNewSet(i)));
                    %                         J0_allsets(i, :) = reshape(J0_curr(:) ./ sqrt(nSource), nSource * nTarg, 1);
                    %
                    %                     end
                    %
                    %                     % hyperbolic tan transformation for uniformness
                    %                     %J_allsets = cell2mat(arrayfun(@(iSet) atanh(J_allsets(i,:)), 1:nSets-1, 'un', false)');
                    %                     J0_resh = reshape(J0_allsets', 1, (nSets - 1) * (nTarg * nSource));
                    %                     J0_setLbl = repelem(allSetIDs(2:end), nTarg*nSource);
                    %                     boxTbl = table(J0_resh, J0_setLbl, 'VariableNames', {'J_vals', 'set_ID'});
                    %                     figure('color','w');
                    %                     set(gcf, 'units', 'normalized', 'outerposition', [0.02 0.1 0.95 0.8])
                    %
                    %                     axB = axes('NextPlot','add','FontSize',14, 'fontweight', 'bold', 'TickDir','out');
                    %                     boxplot( boxTbl.J_vals, boxTbl.set_ID, 'Whisker', 5, 'Parent', axB );
                    %                     boxLines = findobj(axB,'Type','Line');
                    %                     arrayfun( @(x) set(x,'LineStyle','-','Color','k','LineWidth',1), boxLines )
                    %                     boxObjs = findobj(axB,'Tag','Box');
                    %                     arrayfun( @(iBox) patch( boxObjs(iBox).XData, boxObjs(iBox).YData, histGradColors3(iBox, :), 'FaceAlpha', 0.5), 1 : nSets - 1 )
                    %                     grid minor
                    %
                    %
                    %                     colormap cool
                    %                     scatter(boxTbl.set_ID, boxTbl.J_vals, 30, linspace(1, nSets - 1, (nSets - 1) * nTarg * nSource), 'filled', ...
                    %                         'jitter', 'on', 'jitterAmount', 0.3, ...
                    %                         'markerfacealpha', 0.5)
                    %                     grid minor
                    %                     set(axB, 'xlim', [0.5 nSets-0.5])
                    %                     xlabel('# set'),
                    %                     title('J0 distributions')
                    
                    %%
                    
                    if iSource == 1
                        title(AxJ(count), ['#trls=', num2str(length(setTrlInds))], 'fontweight', 'bold', 'fontsize', 12)
                    end
                    
                    if i == 1
                        ylabel([rgns{iSource,1}, ' > '], 'fontweight', 'bold')
                        set(get(gca, 'ylabel'), 'rotation', 0, 'horizontalalignment', 'right')
                    end
                    
                    if count >= length(AxJ) - (nDeltasToPlot) + 1
                        % set(AxJ(count), 'xticklabel', xticksLbl)
                        % xlabel(['set ', num2str(s)], 'fontweight', 'bold')
                        xlabel(['\Delta trls ', num2str(i+1), '&', num2str(i)], 'fontweight', 'bold')
                    end
                    
                    count = count + 1;
                    
                end
            end
            
            % end
            
            
        end
        
        allAxPos = cell2mat(get(AxJ, 'position'));
        oldWidth = mean(unique(allAxPos(:, 3)));
        arrayfun(@(x) set(AxJ(x), 'Position', [allAxPos(x,1), allAxPos(x,2), 1.03*oldWidth, allAxPos(x,4)]), 1:length(AxJ))
        
        arrayfun(@(x) line(AxJ(x), [0 0], get(AxJ(x), 'ylim'), 'linestyle', ':', 'color','black','linewidth', 1), 1:length(AxJ))
        
        titleAxNum = round(0.5*(nSets-1));
        text(AxJ(titleAxNum ), 3.3 * min(get(AxJ(titleAxNum ), 'xlim')), 1.3 * max(get(AxJ(titleAxNum ), 'ylim')), ...
            ['subJ between trial deltas to ',  rgns{iTarget,1}, ' (range: +/- ', num2str(max((3/8) * Jlim)), ')'], 'fontweight', 'bold', 'fontsize', 13)
        
        % set(AxJ(length(AxJ) - (nSets - 1) + 1 : length(AxJ)), 'xtick', linspace(Jlim(1), Jlim(2), 5), 'xticklabel', xticksLbl)
        xlblAxNum = round((nRegions - sum(badRgn))*(nSets - 1) - 0.5*(nSets - 1));
        text(AxJ(xlblAxNum), 1.2 * min(get(AxJ(xlblAxNum), 'xlim')), 1.6 * min(get(AxJ(xlblAxNum), 'ylim')),...
            'log subJ distributions over sets', 'fontweight', 'bold', 'fontsize', 13)
        
        
    end
    
    
    
end

