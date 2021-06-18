clearvars;
close all

% params
a               = 0.01; % idk comes from softNormalize > o3_AddCurrentsToTD_runs
center_data     = true;
n_PCs           = 10;
doSmooth        = true;

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
    
    J = [];
    J0 = [];
    D = cell(nTrls, 1);
    R_ds = cell(nTrls, 1);
    tData = cell(nTrls, 1);
    
    
    for i = 1:nTrls % for each trial in a session
        
        mdlfnm = [mdlDir, currSsn(i).name];
        load(mdlfnm)
        
        mdl             = RNN.mdl;
        J(:, :, i)      = mdl.J;
        J0(:, :, i)     = mdl.J0;
        D{i}            = mdl.targets;
        
        % get indices for each sample of model data
        tData{i}        = mdl.tData;
        R_ds{i}         = mdl.RMdlSample;
        
        if i == 1 % assume or check if same for all
            
            spikeInfoName = [currSsn(i).name(5 : median(strfind(currSsn(i).name, '_'))-1), '_meta.mat'];
            load([spikeInfoPath, spikeInfoName], 'spikeInfo')
            binSize = mdl.dtData; % in sec
            
            % set up indexing vectors for submatrices
            rgns            = mdl.params.arrayRegions;
            arrayList       = rgns(:, 2);
            nRegions        = length(arrayList);
            inArrays        = arrayfun(@(aa) strcmp(spikeInfo.array,arrayList{aa}), 1:nRegions, 'un', false)';
            rgnColors       = brewermap(nRegions, 'Spectral');% cmap(round(linspace(1, 255, nRegions)),:);
        end
        
    end
    
    
    %% semi-convert to matt
    
    td = struct;
    DFull = cell2mat(D');
    nSP = size(DFull,2);
    
    
    for iRgn = 1 : nRegions
        in_rgn = inArrays{iRgn};
        td.([rgns{iRgn,1}, '_spikes']) = DFull(in_rgn, :)';
    end
    
    % get currents (normalized as in softNormalize as far as I can tell...)
    for i = 1 : nTrls
        
        % For each trial
        JTrl = squeeze(J(:, :, i));
        RTrl = R_ds{i};
        
        for iTarget = 1:nRegions
            in_target = inArrays{iTarget};
            
            for iSource = 1:nRegions
                in_source = inArrays{iSource};
                
                if sum(in_target) >= n_PCs && sum(in_source) >= n_PCs
                    
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
                    
                    if doSmooth
                        P_both = smooth_data(P_both', binSize, binSize*10)';
                        P_inh = smooth_data(P_inh', binSize, binSize*10)';
                        P_exc = smooth_data(P_exc', binSize, binSize*10)';
                    end
                    
                    % normalize after if smoothing
                    P_both = P_both ./ (range(P_both,2) + a);
                    P_inh = P_inh ./ (range(P_inh,2) + a);
                    P_exc = P_exc ./ (range(P_exc,2) + a);
                    
                    % PCA for combined currents from all trial types within a given source and
                    % region
                    [w, scores, eigen,~,~,mu] = pca(P_both', 'Algorithm','svd', 'Centered', center_data, 'Economy',0);
                    td.(['Curr' rgns{iSource,1}, rgns{iTarget,1}]) = P_both';
                    
                    % combined only: project into low-D space and pick the num_dims requested
                    projData = P_both(:, :)'; % # sample points x neurons for a given stim
                    temp_proj = (projData - repmat(mu,size(projData,1),1)) * w;
                    td.(['Curr' rgns{iSource,1}, rgns{iTarget,1}, '_pca']) = temp_proj;
                    
                    % norm of n_PCs requested
                    top_proj = temp_proj(:, 1:n_PCs);
                    norm_proj = zeros(size(top_proj,1),1);
                    for tt = 1:size(top_proj,1)
                        norm_proj(tt) = norm(top_proj(tt, :));
                    end
                    
                    norm_projAll{i} = norm_proj;
                    P_inhAll{i} = P_inh(:, :)';
                    P_excAll{i} = P_exc(:, :)';
                    
                    
                end
            end
        end
    end
    
    td.(['Curr' rgns{iSource,1}, rgns{iTarget,1}, '_pca_norm']) = norm_projAll;
    td.(['Curr' rgns{iSource,1}, rgns{iTarget,1},'_inh']) = P_inhAll;
    td.(['Curr' rgns{iSource,1}, rgns{iTarget,1},'_exc']) = P_excAll;
    
    
    %% target activity by region
    
    % set up figure
    figure('color','w');
    AxD = arrayfun(@(i) subplot(2,4,i,'NextPlot', 'add', 'Box', 'on', 'TickDir','out', 'FontSize', 12, 'Fontweight', 'bold',  ...
        'xtick', '', 'ytick', ''), 1:nRegions);
    cm = brewermap(100,'*RdGy');
    
    softFac = 0.0005;
    
    for iRgn = 1 : nRegions
        
        rgnLabel = rgns{iRgn, 1};
        rgnLabel(strfind(rgnLabel, '_')) = ' ';
        
        a = td.([rgns{iRgn,1}, '_spikes']);
        [~, idx] = max(a, [], 1);
        [~, rgnSort] = sort(idx);
        
        bad_units = mean(a,1) == 0;
        
        normfac = mean(abs(a(:, ~bad_units)), 1);
        tmpRgn = a./ (normfac + softFac);
        
        subplot(2, 4, iRgn),
        imagesc(tmpRgn(:, rgnSort)' )
        title(rgnLabel, 'fontweight', 'bold')
        set(gca, 'xcolor', rgnColors(iRgn,:),'ycolor', rgnColors(iRgn,:), 'linewidth', 2)
        
        if iRgn == 3
            text(gca, -0.75*mean(get(gca,'xlim')), 1.1*max(get(gca,'ylim')), ...
                'target rates', 'fontweight', 'bold', 'fontsize', 13)
        end
        
    end
    
    
    axis(AxD, 'tight'),
    set(gcf, 'units', 'normalized', 'outerposition', [0.05 0.1 0.9 0.8])
    
    oldpos = get(AxD(4),'Position');
    colorbar(AxD(4)), colormap(cm);
    set(AxD(4),'Position', oldpos)
    print('-dtiff', '-r400', [RNNfigdir, 'targets_by_rgn_', RNNname])
    close
    
    %% STOPPING HERE ALSO IN FUTURE LUMP TOGETHER SETS
    %% D) CURBD of activity in each region for the juice trials.
    % Left heatmaps show the full Model RNN activity. Remaining heatmaps show
    % the decomposition for each of the sixteen source currents capturing all
    % possible inter-region interactions
    
    
    
    %% plot currents (imagesc)
    close all
    cm = brewermap(100,'*RdBu');
    c = [-3.5 3.5];
    
    for iTarget = 1:nRegions % One plot per target
        
        % sort on CurrTARGTARG in water for some reason
        a = td(2).(['Curr', rgns{iTarget,1}, rgns{iTarget,1}]);
        a = a./repmat(mean(abs(a),1),size(a,1),1);
        [~,idx] = max(a,[],1);
        [~,idx] = sort(idx);
        
        figure('color','w');
        Ax4 = arrayfun(@(i) subplot(3,3,i,'NextPlot', 'add', 'Box', 'off',  'TickDir', 'out', 'FontSize', 10, ...
            'xtick', '', 'ytick', '', 'CLim', c), 1:nRegions^2);
        count = 1;
        for iStim = 1:nStim % Rows are stim
            
            for iSource = 1:nRegions % Cols are sources
                subplot(nStim,nRegions,count);
                
                % divide by mean(abs(val)) of current for each unit
                P = td(iStim).(['Curr', rgns{iSource,1}, rgns{iTarget,1}]);
                P = P(:,idx); P = P ./ mean(abs(P),1);
                
                imagesc(P');
                title([rgns{iSource,1} ' > ' rgns{iTarget,1}],'FontWeight', 'bold')
                ylabel(td(iStim).type,'fontweight', 'bold')
                
                if count==2
                    text(gca, (2/3)*mean(get(gca,'xlim')), 1.2*max(get(gca,'ylim')), ...
                        'currents', 'fontweight', 'bold', 'fontsize', 13)
                end
                
                count = count + 1;
                
            end
        end
        
        axis(Ax4, 'tight'),
        colormap(cm);
        set(gcf, 'units', 'normalized', 'outerposition', [0.25 0 0.4 0.8])
        print('-dtiff', '-r400', [RNNfigdir, 'currents_to_', rgns{iTarget,1}, '_', RNNname(1:end-4)])
        
    end
    
    
    %% plot norm of projection of top n_PCs (from all trials) of currents(line) - THIS IS THE PROBLEM - NORM OF THE TOP 10 PCs!
    close all
    
    for iTarget = 1:nRegions % One plot per target
        
        figure('color','w');
        Ax3 = arrayfun(@(i) subplot(3,3,i,'NextPlot', 'add', 'Box', 'off',  'TickDir', 'out', 'FontSize', 10, ...
            'xtick', ''), 1:nRegions^2);
        count = 1;
        for iSource = 1:nRegions % Rows are sources
            
            for iStim = 1:nStim % Cols are stim
                subplot(nRegions,nStim,count);
                
                P = td(iStim).(['Curr' rgns{iSource,1}, rgns{iTarget,1}, '_pca_norm']);
                
                P = P - repmat(P(1,:),size(P,1),1);
                plot(P(:,1),'LineWidth',2,'Color', rgnColors(iSource,:))
                
                if iSource == 1
                    title(td(iStim).type,'fontweight', 'bold')
                end
                
                if iStim == 1
                    ylabel([rgns{iSource,1} ' > ' rgns{iTarget,1}],'FontWeight', 'bold')
                end
                count = count + 1;
                
            end
        end
        
        axis(Ax3, 'tight'),
        ymin = round(min(min(cell2mat(get(Ax3,'ylim')))),2,'decimals');
        ymax = round(max(max(cell2mat(get(Ax3,'ylim')))),2,'decimals');
        set(Ax3,'ylim', [ymin ymax])
        arrayfun(@(s) line(Ax3(s), get(Ax3(s),'xlim'), [0 0], 'linestyle', ':', 'color','black','linewidth', 1.5), 1:length(Ax3))
        
        print('-dtiff', '-r400', [RNNfigdir, 'pca_normed_currents_to_', rgns{iTarget,1}, '_', RNNname(1:end-4)])
        
    end
    
    %% F. Magnitude of bidirectional currents from striatum to ACC (red, top)
    % and ACC to striatum (yellow, bottom) during presentation of the three
    % stimuli. Solid line: Monkey D; dashed line: Monkey H. Error bars:
    % standard deviation across five different random initializations of the
    % Model RNNs. Schematics (top row) summarize the dominant source currents
    % inferred by CURBD?magnitude and directionality?between the two regions.
    
    % TO DO: Get CURBD from striatum to ACC (red for striatum) and from ACC to
    % striatum (yellow) over all stimuli
    allEndT = [find(mdl.tData<=1.5,1,'last'), find(mdl.tData<=3,1,'last'), find(mdl.tData<=4.5,1,'last')] ;
    
    CurrVStoSC = J(in_SC, in_VS) * R_ds(in_VS, :);
    CurrSCtoVS = J(in_VS, in_SC) * R_ds(in_SC, :);
    
    % CURBD{iTarget,iSource}
    % from striatum to ACC
    % from 3 to 2 aka from VS to SC...will have nUnits = nUnits in target region
    figure,
    subplot(2,1,1),
    plot(fliplr(mean(abs(CurrVStoSC),1)), 'linewidth', 2.5, 'color', rgnColors(3,:))
    line(gca,[allEndT(1), allEndT(1)], [min(get(gca,'YLim')) max(get(gca,'YLim'))], 'color', 'black', 'linewidth', 1, 'linestyle', '--')
    line(gca,[allEndT(2), allEndT(2)], [min(get(gca,'YLim')) max(get(gca,'YLim'))], 'color', 'black', 'linewidth', 1, 'linestyle', '--')
    title('Curr VS > SC')
    set(gca,'xtick', allEndT - 0.5*mean(diff(allEndT)) , 'xticklabel', stimNames)
    
    % from ACC to striatum
    subplot(2,1,2),
    plot(fliplr(mean(abs(CurrSCtoVS),1)), 'linewidth', 2.5, 'color', rgnColors(2,:))
    line(gca,[allEndT(1), allEndT(1)], [min(get(gca,'YLim')) max(get(gca,'YLim'))], 'color', 'black', 'linewidth', 1, 'linestyle', '--')
    line(gca,[allEndT(2), allEndT(2)], [min(get(gca,'YLim')) max(get(gca,'YLim'))], 'color', 'black', 'linewidth', 1, 'linestyle', '--')
    title('Curr SC > VS')
    set(gca,'xtick', allEndT - 0.5*mean(diff(allEndT)) , 'xticklabel', stimNames)
    print('-dtiff', '-r400', [RNNfigdir, 'F_', RNNname(1:end-4)])
    close
    
    c = [-3.5 3.5];
    cm = brewermap(100,'*RdBu');
    
    a = 0.01;
    
    for iTarget = 1 %:nRegions
        
        in_target = in_Rgn{iTarget};
        
        figure('color','w');
        Ax4 = arrayfun(@(i) subplot(3,3,i,'NextPlot', 'add', 'Box', 'on', 'FontSize', 10, ...
            'xtick', '', 'ytick', ''), 1:nRegions^2);
        count = 1;
        
        for iSource = 1:nRegions
            in_source = in_Rgn{iSource};
            
            P1 = J(in_target, in_source) * R_ds(in_source, :);
            
            normfac_juice = range(P1,2) + a; % from softNormalize
            normfac_water = range(P2,2) + a; % from softNormalize
            normfac_cs = range(P3,2) + a; % from softNormalize
            
            % sort on juice sort to from region to itself
            tmpP = J(in_target, in_target) * R_ds(in_target,:);
            tmpP = tmpP(:, in_Stim{3}); % in juice
            tmpP = tmpP ./ mean(abs(tmpP),2); % divide by mean over trial type for each unit (
            [~,iSort] = max(tmpP,[],2);
            [~,iSort] = sort(iSort);
            
            for iStim = 1:nStim
                during_stim = in_Stim{iStim};
                
                % TO DO: question - divide by mean within trial type OR over ALL trial
                % types???? i guess within since........different models
                % essentially......????????? idk, how do reset points work with
                % training the Js?
                
                P_current = P1(:, during_stim);
                P_current = P_current ./ normfac_juice;
                P_current = P_current(iSort, :);
                
                % divide by the mean of the absolute value of current for each unit
                P_plot = P_current ./ mean(abs(P_current),2);
                
                
                subplot(nRegions,nStim,count); hold all; count = count + 1;
                
                % plot(P_plot, 'linewidth', 2, 'color', hist_colors(iSource,:))
                
                imagesc(P_plot(iSort,:))
                axis tight;
                set(gca,'Box','off','TickDir','out','FontSize',10);
                set(gca,'CLim',c);
                % line(gca,[min(get(gca,'XLim')) max(get(gca,'XLim'))], [0 0], 'color', 'black', 'linewidth', 1, 'linestyle', '--')
                % title(['Curr ', regions{iSource,1} ' > ' regions{iTarget,1}]);
                
                if iSource == iTarget
                    title(stimNames{iStim},'FontWeight','Bold')
                end
                
                if iStim ==1
                    ylabel([rgns{iSource,1} ' > ' rgns{iTarget,1}],'FontWeight', 'bold')
                end
            end
        end
        
        
        set(Ax4,'ylim', [min(cellfun(@min, get(Ax4,'ylim'))), max(cellfun(@max, get(Ax4, 'ylim')))] )
        colormap(cm);
        print('-dtiff', '-r400', [RNNfigdir, 'mean_abs_currents_to_', rgns{iTarget,1}, '_', RNNname(1:end-4)])
        
    end
    
    close
    
end



%% histograms of subJs

reshapedJplot = reshape(J,sum(nUnits)^2,1)./sqrt(sum(nUnits));
reshapedJ0plot = reshape(J0,sum(nUnits)^2,1)./sqrt(sum(nUnits));
maxabsval = 1.02*max(abs([reshapedJplot;reshapedJ0plot]));
Jlim = [-maxabsval, maxabsval];
binwidth = (Jlim(2)-Jlim(1))/100;
[bincounts,edgesnew] = histcounts(reshapedJplot,Jlim(1):binwidth:Jlim(2));

histcenters = edgesnew(1:end-1) + (diff(edgesnew) ./ 2);
cmax = max(abs(([J(:)./sqrt(sum(nUnits));J0(:)./sqrt(sum(nUnits))])));




figure('color','w');
AxD = arrayfun(@(i) subplot(3,3,i,'NextPlot', 'add', 'Box', 'off'), 1:nRegions^2);
% hold on,

% pjns to AMY
J_amy_to_amy = J(in_AMY,in_AMY);
J_sc_to_amy = J(in_AMY, in_SC);
J_vs_to_amy = J(in_AMY, in_VS);

% ./sqrt(nUnits in source)
[J_amy_to_amy_N] = histcounts(reshape(J_amy_to_amy(:)./sqrt(nAMY),nAMY^2,1),Jlim(1):binwidth:Jlim(2)); % sqrt(nAMY)*
[J_sc_to_amy_N] = histcounts(reshape(J_sc_to_amy(:)./sqrt(nSC),nSC*nAMY,1),Jlim(1):binwidth:Jlim(2)); % sqrt(nSC)*
[J_vs_to_amy_N] = histcounts(reshape(J_vs_to_amy(:)./sqrt(nVS),nVS*nAMY,1),Jlim(1):binwidth:Jlim(2)); % sqrt(nVS)*

subplot(3,3,1), semilogy(histcenters,J_amy_to_amy_N./max(J_amy_to_amy_N), 'o-', 'color', rgnColors(1,:)),  title('AMY to AMY')
subplot(3,3,2), semilogy(histcenters,J_sc_to_amy_N./max(J_sc_to_amy_N), '*-', 'color', rgnColors(1,:)), title('SC to AMY')
subplot(3,3,3), semilogy(histcenters,J_vs_to_amy_N./max(J_vs_to_amy_N), 'v-', 'color', rgnColors(1,:)), title('VS to AMY')

% pjns to SC
J_amy_to_sc = J(in_SC, in_AMY);
J_sc_to_sc = J(in_SC, in_SC);
J_vs_to_sc = J(in_SC, in_VS);

[J_amy_to_sc_N] = histcounts(reshape(J_amy_to_sc(:)./sqrt(nAMY),nAMY*nSC,1),Jlim(1):binwidth:Jlim(2)); % sqrt(nAMY)*
[J_sc_to_sc_N] = histcounts(reshape(J_sc_to_sc(:)./sqrt(nSC),nSC^2,1),Jlim(1):binwidth:Jlim(2)); % sqrt(nSC)*
[J_vs_to_sc_N] = histcounts(reshape(J_vs_to_sc(:)./sqrt(nVS),nVS*nSC,1),Jlim(1):binwidth:Jlim(2)); % sqrt(nVS)*

subplot(3,3,4), semilogy(histcenters,J_amy_to_sc_N./max(J_amy_to_sc_N), 'o-', 'color', rgnColors(2,:)), title('AMY to SC')
subplot(3,3,5), semilogy(histcenters,J_sc_to_sc_N./max(J_sc_to_sc_N), '*-', 'color', rgnColors(2,:)), title('SC to SC')
subplot(3,3,6), semilogy(histcenters,J_vs_to_sc_N./max(J_vs_to_sc_N), 'v-', 'color', rgnColors(2,:)), title('VS to SC')

% pjns to VS

J_amy_to_vs = J(in_VS, in_AMY);
J_sc_to_vs = J(in_VS, in_SC);
J_vs_to_vs = J(in_VS, in_VS);

[J_amy_to_vs_N] = histcounts(reshape(J_amy_to_vs(:)./sqrt(nAMY),nAMY*nVS,1),Jlim(1):binwidth:Jlim(2)); % *sqrt(nAMY)
[J_sc_to_vs_N] = histcounts(reshape(J_sc_to_vs(:)./sqrt(nSC),nSC*nVS,1),Jlim(1):binwidth:Jlim(2)); % sqrt(nSC)*
[J_vs_to_vs_N] = histcounts(reshape(J_vs_to_vs(:)./sqrt(nVS),nVS^2,1),Jlim(1):binwidth:Jlim(2)); % sqrt(nVS)*

subplot(3,3,7), semilogy(histcenters,J_amy_to_vs_N./max(J_amy_to_vs_N), 'o-', 'color', rgnColors(3,:)), title('AMY to VS')
subplot(3,3,8), semilogy(histcenters,J_sc_to_vs_N./max(J_sc_to_vs_N), '*-', 'color', rgnColors(3,:)), title('SC to VS')
subplot(3,3,9), semilogy(histcenters,J_vs_to_vs_N./max(J_vs_to_vs_N), 'v-', 'color', rgnColors(3,:)), title('VS to VS')

arrayfun(@(x) set(AxD(x),'xlim', [-1.25*cmax 1.25*cmax], 'YSCale', 'log', 'ylim', [0.0001 1]), 1:length(AxD));
AxL = findobj(gcf,'Type','Line');
arrayfun(@(x) set(AxL(x), 'linewidth', 1.5, 'markersize', 2), 1:length(AxL));

% TO DO: SET TO PRINT

