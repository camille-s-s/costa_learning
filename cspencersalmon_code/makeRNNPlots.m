clearvars;
close all

% params
aFac            = 0.01; % idk comes from softNormalize > o3_AddCurrentsToTD_runs
center_data     = true;
n_PCs           = 10;
doSmooth        = true;
rgnOrder        = [1 8 2 7 4 5 6 3]; % for 4 row 2 col plots

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
    nSets = length(allSetIDs);
    % nTrls = 1;
    
    J = [];
    J0 = [];
    D = cell(nTrls, 1);
    R_ds = cell(nTrls, 1);
    tData = cell(nTrls, 1);
    setID = [];
    
    for i = 1 : nTrls % for each trial in a session
        
        mdlfnm = [mdlDir, currSsn(i).name];
        load(mdlfnm)
        
        mdl             = RNN.mdl;
        J(:, :, i)      = mdl.J;
        J0(:, :, i)     = mdl.J0;
        D{i}            = mdl.targets;
        setID(i)        = mdl.setID;
        
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
            % inArrays        = arrayfun(@(aa) strcmp(spikeInfo.array,arrayList{aa}), 1:nRegions, 'un', false)';
            rgnColors       = brewermap(nRegions, 'Spectral');% cmap(round(linspace(1, 255, nRegions)),:);
        end
        
    end
    
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
                    % Interp for PCA
                    % RTrl = cell2mat(arrayfun(@(n) interp(RTrl(n, :),10), 1 : size(RTrl, 1), 'un', false)'); % changes dtData to dtData/10 (in
                    % seconds)
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
    
    % prune bad units by cutting any unit whose overall FR is below threshold
    
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
    
    %% target activity by region
    
    % set up figure
    figure('color','w');
    set(gcf, 'units', 'normalized', 'outerposition', [0 0 1 1])
    
    %     AxTargs = arrayfun(@(i) subplot(4,2,i,'NextPlot', 'add', 'Box', 'on', 'TickDir','out', 'FontSize', 10, 'Fontweight', 'bold',  ...
    %         'xtick', newTrlInds(1:2:end), 'xticklabel', 1:2:nTrls, 'ytick', ''), 1:nRegions);
    AxTargs = arrayfun(@(i) subplot(4,2,i,'NextPlot', 'add', 'Box', 'on', 'TickDir','out', 'FontSize', 10, 'Fontweight', 'bold',  ...
        'xtick', 1:50:shortestTrl, 'xticklabel', dtData*((1:50:shortestTrl) - 1), 'ytick', ''), 1:nRegions);
    cm = brewermap(100,'*RdGy');
    
    softFac = 0.0000;
%     for iRgn =  1 : nRegions
%         if ~ismember(iRgn, find(badRgn))
%             rgnLabel = rgns{iRgn, 1};
%             rgnLabel(strfind(rgnLabel, '_')) = ' ';
%             
%             a = td.([rgns{iRgn,1}, '_spikes_avg']); % T x N
%             
%             % get the sort
%             [~, idx] = max(a, [], 1);
%             [~, rgnSort] = sort(idx);
%             
%             normfac = mean(abs(a), 1); % 1 x N
%             tmpRgn = a; %./ (normfac + softFac);
%             
%             subplot(AxTargs(iRgn))
%             imagesc(tmpRgn(:, rgnSort)' )
%             axis(AxTargs(iRgn), 'tight')
%             
%             %             if min(firstTrlNewSet) <= max(newTrlInds)
%             %                 for s = 1:size(firstTrlNewSet, 2)
%             %                     line(gca, [newTrlInds(firstTrlNewSet(s)), newTrlInds(firstTrlNewSet(s))], get(gca, 'ylim'), 'color', 'green', 'linewidth', 1.5)
%             %                     if s ~= length(firstTrlNewSet)
%             %                         setLabelPos = 0.5 * (newTrlInds(firstTrlNewSet(s)) + newTrlInds(firstTrlNewSet(s+1)));
%             %                     else
%             %                         setLabelPos = 0.5 * (newTrlInds(firstTrlNewSet(s)) + length(DFull));
%             %                     end
%             %                     text(gca, setLabelPos, -5, ['set ', num2str(allSetIDs(s + 1))], 'fontweight', 'bold')
%             %                 end
%             %             end
%             
%             title(rgnLabel, 'fontweight', 'bold')
%             set(gca, 'xcolor', rgnColors(iRgn,:),'ycolor', rgnColors(iRgn,:), 'linewidth', 2)
%             xlabel('time (sec)'), ylabel('neurons')
%             
%             if iRgn == 2
%                 text(gca, -0.75*mean(get(gca,'xlim')), 1.1*max(get(gca,'ylim')), ...
%                     [allFiles{f}, ' trial averaged target rates'], 'fontweight', 'bold', 'fontsize', 13)
%                 [oldxlim, oldylim] = size(tmpRgn);
%             end
%         end
%     end
%     
%     set(AxTargs, 'CLim', [0 1])
%     oldpos = get(AxTargs(2),'Position');
%     colorbar(AxTargs(2)), colormap(cm);
%     set(AxTargs(2),'Position', oldpos)
%     set(AxTargs(2), 'xlim', [0 oldxlim], 'ylim', [0 oldylim])
%     print('-dtiff', '-r400', [RNNfigdir, allFiles{f}, '_trl_avged_targets_by_rgn'])
%     close
%     
    %% STOPPING HERE ALSO IN FUTURE LUMP TOGETHER SETS
    %% D) CURBD of activity in each region for the juice trials.
    % Left heatmaps show the full Model RNN activity. Remaining heatmaps show
    % the decomposition for each of the sixteen source currents capturing all
    % possible inter-region interactions
    
    
    
    %% plot currents (imagesc) - to do: currents over time or currents averaged?
    close all
    cm = brewermap(100,'*RdBu');
    c = [-3.5 3.5];
    
    %     for iTarget = 1:nRegions % One plot per target
    %         in_target = inArrays{iTarget};
    %
    %         if ~ismember(iTarget, find(badRgn))
    %             a = td.(['Curr', rgns{iTarget,1}, '_', rgns{iTarget,1}]);
    %             a = a./repmat(mean(abs(a), 1), size(a, 1), 1);
    %             [~,idx] = max(a, [], 1);
    %             [~,idx] = sort(idx);
    %         else
    %             continue
    %         end
    %
    %         figure('color','w');
    %         AxCurr = arrayfun(@(i) subplot(4,2,i,'NextPlot', 'add', 'Box', 'off',  'TickDir', 'out', 'FontSize', 10, 'Fontweight', 'bold',...
    %             'xtick', newTrlInds(1:2:end), 'xticklabel', 1:2:nTrls, 'ytick', '', 'CLim', c), 1:nRegions);
    %         count = 1;
    %
    %         for iSource = 1:nRegions
    %             in_source = inArrays{iSource};
    %
    %             if sum(in_target) >= n_PCs && sum(in_source) >= n_PCs
    %
    %                 subplot(AxCurr(iSource));
    %                 % divide by mean(abs(val)) of current for each unit
    %                 P = td.(['Curr', rgns{iSource,1}, '_', rgns{iTarget,1}]);
    %                 P = P(:,idx); P = P ./ mean(abs(P),1);
    %
    %                 imagesc(P');
    %
    %                 if min(firstTrlNewSet) <= max(newTrlInds)
    %                     for s = 1:length(firstTrlNewSet)
    %                         line(gca, [newTrlInds(firstTrlNewSet(s)), newTrlInds(firstTrlNewSet(s))], get(gca, 'ylim'), 'color', 'black', 'linewidth', 1.5)
    %                         if s ~= length(firstTrlNewSet)
    %                             setLabelPos = 0.5 * (newTrlInds(firstTrlNewSet(s)) + newTrlInds(firstTrlNewSet(s+1)));
    %                         else
    %                             setLabelPos = 0.5 * (newTrlInds(firstTrlNewSet(s)) + length(DFull));
    %                         end
    %                         text(gca, setLabelPos, -15, ['set ', num2str(allSetIDs(s + 1))], 'fontweight', 'bold')
    %                     end
    %                 end
    %
    %                 title([rgns{iSource,1} ' > ' rgns{iTarget,1}],'FontWeight', 'bold')
    %                 xlabel('trials'), ylabel('neurons')
    %
    %                 if count==2
    %                     text(gca, -(1/2) * mean(get(gca,'xlim')), 1.2 * max(get(gca,'ylim')), ...
    %                         [allFiles{f}, ' currents to ', rgns{iTarget, 1}], 'fontweight', 'bold', 'fontsize', 13)
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
    %         axis(AxCurr, 'tight'),
    %         colormap(cm);
    %         set(gcf, 'units', 'normalized', 'outerposition', [0.05 0.1 1 0.9])
    %         oldpos = get(AxCurr(2),'Position');
    %         colorbar(AxCurr(2)), colormap(cm);
    %         set(AxCurr(2),'Position', oldpos)
    %         set(AxCurr(2), 'xlim', [0 oldxlim], 'ylim', [0 oldylim])
    %
    %         print('-dtiff', '-r400', [RNNfigdir, allFiles{f}, '_curr_to_', rgns{iTarget,1}])
    %
    %     end
    
    
    %% plot norm of projection of top n_PCs (from all trials) of currents(line) - THIS IS THE PROBLEM - NORM OF THE TOP 10 PCs!
    close all
    
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
    %
    %     end
    
    
    %% HISTOGRAMS OF SUBJs
    
    cm = colormap(cool);
    histGradColors = cm(round(linspace(1, size(cm, 1), nTrls - 1)),:);
    nHistBins = 100;
    
    for iTarget = 1:nRegions % two plots per target
        
        in_target = inArrays{iTarget};
        nTarg = sum(in_target);
        
        Jtarg = squeeze(J(in_target, in_target, :));
        maxabsval = round(1.02 * max(abs((Jtarg(:)./sqrt(nTarg)))), 2, 'decimals');
        
        Jlim = [-maxabsval, maxabsval]; % same JLim for each target no matter the source
        histBinWidth = (Jlim(2) - Jlim(1))/nHistBins;
        edgesnew = linspace(Jlim(1), Jlim(2), nHistBins + 1);
        histcenters = edgesnew(1:end-1) + (diff(edgesnew) ./ 2);
        
        % LEFT
        count = 1;
        figure('color','w');
        set(gcf, 'units', 'normalized', 'outerposition', [0 0 1 1])
        
        AxJ = arrayfun(@(i) subplot(0.5 * nRegions, nSets - 1, i,'NextPlot', 'add', 'Box', 'off', ...
            'yscale', 'log', 'ylim', [0.001 1], 'xlim', 5/12 * Jlim, 'xtick', ''), 1:((0.5 * nRegions) * (nSets-1)));
               
        for iSource = [1 3 5 7] % 1:nRegions
            
            in_source = inArrays{iSource};
            nSource = sum(in_source);
            
            if sum(in_target) >= n_PCs && sum(in_source) >= n_PCs
                                
                for s = allSetIDs(2 : end)
                    setTrlInds = find(setID == s);
                    subJ_N = NaN(length(setTrlInds), nHistBins);
                    
                    if exist('colororder','file') ~= 0
                        histGradColors2 = cm(round(linspace(1, size(cm, 1), length(setTrlInds))),:);
                        
                    end
                    % get currents (normalized as in softNormalize as far as I can tell...)
                    for i = 1 : length(setTrlInds)% 1 : nTrls - 1
                        
                        % For each trial in a set, collect Js....
                        subJ_curr = squeeze(J(in_target, in_source, setTrlInds(i)));
                        % subJ_next = squeeze(J(in_target, in_source, i + 1));
                        
                        [subJ_curr_N, ~] = histcounts(reshape(subJ_curr(:) ./ sqrt(nSource), nSource * nTarg, 1), ...
                            Jlim(1) : histBinWidth : Jlim(2));
                        % [subJ_next_N, ~] = histcounts(reshape(subJ_next(:) ./ sqrt(nSource), nSource * nTarg, 1), ...
                        % Jlim(1) : histBinWidth : Jlim(2));
                        
                        subJ_curr_N = subJ_curr_N ./ max(subJ_curr_N);
                        subJ_N(i, :) = subJ_curr_N;
                        
                        % subJ_next_N = subJ_next_N ./ max(subJ_next_N);
                        % subJ_diff = subJ_next_N - subJ_curr_N;
                        % subplot(AxJ(count)), plot(histcenters, subJ_diff, '-', 'linewidth', 1.5, 'color',  histGradColors(i, :)),
                        % subplot(AxJ(count)), plot(histcenters, subJ_diff, '-', 'linewidth', 1.5, 'color',  histGradColors(i, :)),
                    end
                    
                    subplot(AxJ(count)), semilogy(histcenters,subJ_N, 'o-', 'linewidth', 1, 'markersize', 2)
                    colororder(histGradColors2);
                    
                    if s == 1
                        ylabel([rgns{iSource,1}, ' > '], 'fontweight', 'bold')
                        set(get(gca, 'ylabel'), 'rotation', 0, 'horizontalalignment', 'right')
                    end
                    
                    if iSource == nRegions || iSource == nRegions - 1
                        % xlabel([num2str(i+1), '\Delta', num2str(i)], 'fontweight', 'bold')
                        xlabel(['set ', num2str(s)], 'fontweight', 'bold')
                    end
                    
                    count = count + 1;
                    
                    % if i == nTrls - 1
                    %     subJ_N(i+1, :) = subJ_next_N;
                    % end
                    
                end
                
            end
            
            
        end
        
        % allYLims = cell2mat(arrayfun(@(i) get(AxJ(i), 'ylim'), 1:(0.5*nRegions * (nTrls-1)), 'un', 0)');
        % newYLims = [0.95 * min(allYLims(:)), 0.95 * max(allYLims(:))];
        % arrayfun(@(x) set(AxJ(x), 'ylim', newYLims), 1:length(AxJ))
        
        % rgnYMins = 0.95*arrayfun(@(x) min(min(allYLims(x:x+nTrls-2, :))), 1:nTrls-1:length(allYLims)-nTrls+2);
        % rgnYMaxs = 0.95*arrayfun(@(x) max(max(allYLims(x:x+nTrls-2, :))), 1:nTrls-1:length(allYLims)-nTrls+2);
        
        rgnFirstTrl=1:nTrls-1:length(allYLims)-nTrls+2;
        % arrayfun(@(r) set(AxJ(rgnFirstTrl(r):rgnFirstTrl(r)+nTrls-2), 'ylim', [rgnYMins(r) rgnYMaxs(r)]), 1:length(rgnFirstTrl))
        % arrayfun(@(r) set(AxJ(rgnFirstTrl(r)), ...
            % 'ytick', [rgnYMins(r) 0 rgnYMaxs(r)], 'yticklabel', ...
            % [rgnYMins(r), 0, rgnYMaxs(r)], 'fontweight','bold'), 1:length(rgnFirstTrl))
        allAxPos = cell2mat(get(AxJ, 'position'));
        oldWidth = mean(unique(allAxPos(:, 3)));
        arrayfun(@(x) set(AxJ(x), 'Position', [allAxPos(x,1), allAxPos(x,2), 1.03*oldWidth, allAxPos(x,4)]), 1:length(AxJ))
        
        arrayfun(@(x) line(AxJ(x), [0 0], get(AxJ(x), 'ylim'), 'linestyle', ':', 'color','black','linewidth', 1), 1:length(AxJ))
        % arrayfun(@(x) line(AxJ(x), get(AxJ(x), 'xlim'), [0 0], 'linestyle', ':', 'color','black','linewidth', 1), 1:length(AxJ))
        
        titleAxNum = round(0.5*(length(setTrlsInds)));
        text(AxJ(titleAxNum ), 3.3 * min(get(AxJ(titleAxNum ), 'xlim')), 1.3 * max(get(AxJ(titleAxNum ), 'ylim')), ...
            ['subJ between trial deltas to ',  rgns{iTarget,1}, ' (range: +/- ', num2str(max(5/12 * Jlim)), ')'], 'fontweight', 'bold', 'fontsize', 13)
        
        xlblAxNum = round(0.5*nRegions*(length(setTrlsInds)) - 0.5*(length(setTrlsInds)));
        % text(AxJ(xlblAxNum), 1.2 * min(get(AxJ(xlblAxNum), 'xlim')), 1.6 * min(get(AxJ(xlblAxNum), 'ylim')),...
            % '\Delta J values between trials', 'fontweight', 'bold', 'fontsize', 13)
          text(AxJ(xlblAxNum), 1.2 * min(get(AxJ(xlblAxNum), 'xlim')), 1.6 * min(get(AxJ(xlblAxNum), 'ylim')),...
            '\log subJ distributions over sets', 'fontweight', 'bold', 'fontsize', 13)
        
        
    end
    
    
    
end


%     %% F. Magnitude of bidirectional currents from striatum to ACC (red, top)
%     % and ACC to striatum (yellow, bottom) during presentation of the three
%     % stimuli. Solid line: Monkey D; dashed line: Monkey H. Error bars:
%     % standard deviation across five different random initializations of the
%     % Model RNNs. Schematics (top row) summarize the dominant source currents
%     % inferred by CURBD?magnitude and directionality?between the two regions.
%
%     % TO DO: Get CURBD from striatum to ACC (red for striatum) and from ACC to
%     % striatum (yellow) over all stimuli
%     allEndT = [find(mdl.tData<=1.5,1,'last'), find(mdl.tData<=3,1,'last'), find(mdl.tData<=4.5,1,'last')] ;
%
%     CurrVStoSC = J(in_SC, in_VS) * R_ds(in_VS, :);
%     CurrSCtoVS = J(in_VS, in_SC) * R_ds(in_SC, :);
%
%     % CURBD{iTarget,iSource}
%     % from striatum to ACC
%     % from 3 to 2 aka from VS to SC...will have nUnits = nUnits in target region
%     figure,
%     subplot(2,1,1),
%     plot(fliplr(mean(abs(CurrVStoSC),1)), 'linewidth', 2.5, 'color', rgnColors(3,:))
%     line(gca,[allEndT(1), allEndT(1)], [min(get(gca,'YLim')) max(get(gca,'YLim'))], 'color', 'black', 'linewidth', 1, 'linestyle', '--')
%     line(gca,[allEndT(2), allEndT(2)], [min(get(gca,'YLim')) max(get(gca,'YLim'))], 'color', 'black', 'linewidth', 1, 'linestyle', '--')
%     title('Curr VS > SC')
%     set(gca,'xtick', allEndT - 0.5*mean(diff(allEndT)) , 'xticklabel', stimNames)
%
%     % from ACC to striatum
%     subplot(2,1,2),
%     plot(fliplr(mean(abs(CurrSCtoVS),1)), 'linewidth', 2.5, 'color', rgnColors(2,:))
%     line(gca,[allEndT(1), allEndT(1)], [min(get(gca,'YLim')) max(get(gca,'YLim'))], 'color', 'black', 'linewidth', 1, 'linestyle', '--')
%     line(gca,[allEndT(2), allEndT(2)], [min(get(gca,'YLim')) max(get(gca,'YLim'))], 'color', 'black', 'linewidth', 1, 'linestyle', '--')
%     title('Curr SC > VS')
%     set(gca,'xtick', allEndT - 0.5*mean(diff(allEndT)) , 'xticklabel', stimNames)
%     print('-dtiff', '-r400', [RNNfigdir, 'F_', RNNname(1:end-4)])
%     close
%
%     c = [-3.5 3.5];
%     cm = brewermap(100,'*RdBu');
%
%     a = 0.01;
%
%     for iTarget = 1 %:nRegions
%
%         in_target = in_Rgn{iTarget};
%
%         figure('color','w');
%         AxCurr = arrayfun(@(i) subplot(3,3,i,'NextPlot', 'add', 'Box', 'on', 'FontSize', 10, ...
%             'xtick', '', 'ytick', ''), 1:nRegions^2);
%         count = 1;
%
%         for iSource = 1:nRegions
%             in_source = in_Rgn{iSource};
%
%             P1 = J(in_target, in_source) * R_ds(in_source, :);
%
%             normfac_juice = range(P1,2) + a; % from softNormalize
%             normfac_water = range(P2,2) + a; % from softNormalize
%             normfac_cs = range(P3,2) + a; % from softNormalize
%
%             % sort on juice sort to from region to itself
%             tmpP = J(in_target, in_target) * R_ds(in_target,:);
%             tmpP = tmpP(:, in_Stim{3}); % in juice
%             tmpP = tmpP ./ mean(abs(tmpP),2); % divide by mean over trial type for each unit (
%             [~,iSort] = max(tmpP,[],2);
%             [~,iSort] = sort(iSort);
%
%             for iStim = 1:nStim
%                 during_stim = in_Stim{iStim};
%
%                 % TO DO: question - divide by mean within trial type OR over ALL trial
%                 % types???? i guess within since........different models
%                 % essentially......????????? idk, how do reset points work with
%                 % training the Js?
%
%                 P_current = P1(:, during_stim);
%                 P_current = P_current ./ normfac_juice;
%                 P_current = P_current(iSort, :);
%
%                 % divide by the mean of the absolute value of current for each unit
%                 P_plot = P_current ./ mean(abs(P_current),2);
%
%
%                 subplot(nRegions,nStim,count); hold all; count = count + 1;
%
%                 % plot(P_plot, 'linewidth', 2, 'color', hist_colors(iSource,:))
%
%                 imagesc(P_plot(iSort,:))
%                 axis tight;
%                 set(gca,'Box','off','TickDir','out','FontSize',10);
%                 set(gca,'CLim',c);
%                 % line(gca,[min(get(gca,'XLim')) max(get(gca,'XLim'))], [0 0], 'color', 'black', 'linewidth', 1, 'linestyle', '--')
%                 % title(['Curr ', regions{iSource,1} ' > ' regions{iTarget,1}]);
%
%                 if iSource == iTarget
%                     title(stimNames{iStim},'FontWeight','Bold')
%                 end
%
%                 if iStim ==1
%                     ylabel([rgns{iSource,1} ' > ' rgns{iTarget,1}],'FontWeight', 'bold')
%                 end
%             end
%         end
%
%
%         set(AxCurr,'ylim', [min(cellfun(@min, get(AxCurr,'ylim'))), max(cellfun(@max, get(AxCurr, 'ylim')))] )
%         colormap(cm);
%         print('-dtiff', '-r400', [RNNfigdir, 'mean_abs_currents_to_', rgns{iTarget,1}, '_', RNNname(1:end-4)])
%
%     end
%
%     close
%
% end
%
%
%
% %% histograms of subJs
%
% reshapedJplot = reshape(J,sum(nUnits)^2,1)./sqrt(sum(nUnits));
% reshapedJ0plot = reshape(J0,sum(nUnits)^2,1)./sqrt(sum(nUnits));
% maxabsval = 1.02*max(abs([reshapedJplot;reshapedJ0plot]));
% Jlim = [-maxabsval, maxabsval];
% binwidth = (Jlim(2)-Jlim(1))/100;
% [bincounts,edgesnew] = histcounts(reshapedJplot,Jlim(1):binwidth:Jlim(2));
%
% histcenters = edgesnew(1:end-1) + (diff(edgesnew) ./ 2);
% cmax = max(abs(([J(:)./sqrt(sum(nUnits));J0(:)./sqrt(sum(nUnits))])));
%
%
%
%
% figure('color','w');
% AxTargs = arrayfun(@(i) subplot(3,3,i,'NextPlot', 'add', 'Box', 'off'), 1:nRegions^2);
% % hold on,
%
% % pjns to AMY
% J_amy_to_amy = J(in_AMY,in_AMY);
% J_sc_to_amy = J(in_AMY, in_SC);
% J_vs_to_amy = J(in_AMY, in_VS);
%
% % ./sqrt(nUnits in source)
% [J_amy_to_amy_N] = histcounts(reshape(J_amy_to_amy(:)./sqrt(nAMY),nAMY^2,1),Jlim(1):binwidth:Jlim(2)); % sqrt(nAMY)*
% [J_sc_to_amy_N] = histcounts(reshape(J_sc_to_amy(:)./sqrt(nSC),nSC*nAMY,1),Jlim(1):binwidth:Jlim(2)); % sqrt(nSC)*
% [J_vs_to_amy_N] = histcounts(reshape(J_vs_to_amy(:)./sqrt(nVS),nVS*nAMY,1),Jlim(1):binwidth:Jlim(2)); % sqrt(nVS)*
%
% subplot(3,3,1), semilogy(histcenters,J_amy_to_amy_N./max(J_amy_to_amy_N), 'o-', 'color', rgnColors(1,:)),  title('AMY to AMY')
% subplot(3,3,2), semilogy(histcenters,J_sc_to_amy_N./max(J_sc_to_amy_N), '*-', 'color', rgnColors(1,:)), title('SC to AMY')
% subplot(3,3,3), semilogy(histcenters,J_vs_to_amy_N./max(J_vs_to_amy_N), 'v-', 'color', rgnColors(1,:)), title('VS to AMY')
%
% % pjns to SC
% J_amy_to_sc = J(in_SC, in_AMY);
% J_sc_to_sc = J(in_SC, in_SC);
% J_vs_to_sc = J(in_SC, in_VS);
%
% [J_amy_to_sc_N] = histcounts(reshape(J_amy_to_sc(:)./sqrt(nAMY),nAMY*nSC,1),Jlim(1):binwidth:Jlim(2)); % sqrt(nAMY)*
% [J_sc_to_sc_N] = histcounts(reshape(J_sc_to_sc(:)./sqrt(nSC),nSC^2,1),Jlim(1):binwidth:Jlim(2)); % sqrt(nSC)*
% [J_vs_to_sc_N] = histcounts(reshape(J_vs_to_sc(:)./sqrt(nVS),nVS*nSC,1),Jlim(1):binwidth:Jlim(2)); % sqrt(nVS)*
%
% subplot(3,3,4), semilogy(histcenters,J_amy_to_sc_N./max(J_amy_to_sc_N), 'o-', 'color', rgnColors(2,:)), title('AMY to SC')
% subplot(3,3,5), semilogy(histcenters,J_sc_to_sc_N./max(J_sc_to_sc_N), '*-', 'color', rgnColors(2,:)), title('SC to SC')
% subplot(3,3,6), semilogy(histcenters,J_vs_to_sc_N./max(J_vs_to_sc_N), 'v-', 'color', rgnColors(2,:)), title('VS to SC')
%
% % pjns to VS
%
% J_amy_to_vs = J(in_VS, in_AMY);
% J_sc_to_vs = J(in_VS, in_SC);
% J_vs_to_vs = J(in_VS, in_VS);
%
% [J_amy_to_vs_N] = histcounts(reshape(J_amy_to_vs(:)./sqrt(nAMY),nAMY*nVS,1),Jlim(1):binwidth:Jlim(2)); % *sqrt(nAMY)
% [J_sc_to_vs_N] = histcounts(reshape(J_sc_to_vs(:)./sqrt(nSC),nSC*nVS,1),Jlim(1):binwidth:Jlim(2)); % sqrt(nSC)*
% [J_vs_to_vs_N] = histcounts(reshape(J_vs_to_vs(:)./sqrt(nVS),nVS^2,1),Jlim(1):binwidth:Jlim(2)); % sqrt(nVS)*
%
% subplot(3,3,7), semilogy(histcenters,J_amy_to_vs_N./max(J_amy_to_vs_N), 'o-', 'color', rgnColors(3,:)), title('AMY to VS')
% subplot(3,3,8), semilogy(histcenters,J_sc_to_vs_N./max(J_sc_to_vs_N), '*-', 'color', rgnColors(3,:)), title('SC to VS')
% subplot(3,3,9), semilogy(histcenters,J_vs_to_vs_N./max(J_vs_to_vs_N), 'v-', 'color', rgnColors(3,:)), title('VS to VS')
%
% arrayfun(@(x) set(AxTargs(x),'xlim', [-1.25*cmax 1.25*cmax], 'YSCale', 'log', 'ylim', [0.0001 1]), 1:length(AxTargs));
% AxL = findobj(gcf,'Type','Line');
% arrayfun(@(x) set(AxL(x), 'linewidth', 1.5, 'markersize', 2), 1:length(AxL));
%
% % TO DO: SET TO PRINT
%
