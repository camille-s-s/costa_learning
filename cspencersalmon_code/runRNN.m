
clearvars;
close all
% multiRNN = fitRNN(struct( ...
%     'dataset', 'raw_matched_trls', ...
%     'normByRegion', false, ...
%     'plotStatus', true, ...
%     'smoothFlag', false, ...
%     'saveMdl', true));

% params
a = 0.01; % idk comes from softNormalize > o3_AddCurrentsToTD_runs 
center_data = true;
n_PCs = 10;
doSmooth = true;

% in and outdirs
RNNfigdir       = '~/Dropbox (BrAINY Crew)/csRNN/figures/';
RNNname         = 'D_allrgn_1_stimOn_1500_20_150';

mdlfnm = ['~/Dropbox (BrAINY Crew)/csRNN/data/', RNNname, '.mat']; % D_allrgn_1_stimOn_1500_20_150.mat';
load(mdlfnm)

mdl = phrRNN.all(end).mdl;
rgns = mdl.regions;
J = mdl.J;
J0 = mdl.J0;
binSize = mdl.params.Lsec/mdl.params.Lneural; % in sec

% get indices for each sample of model data
R = mdl.RNN; % TO DO: DO YOU SMOOTH R?
tData = mdl.tData;
t = mdl.tRNN;

R_ds = zeros(size(R,1),length(tData));
for ii = 1:size(R,1)
    R_ds(ii,:) = interp1(t,R(ii,:),tData);
end

if isfield(mdl.params,'smoothFlag')
    D = mdl.target_rates; % these if there is smooth_flag will come smoothed
end

nSP = size(D,2);

% set up indexing vectors for submatrices
in_juice = false(1, nSP);
in_juice(1:nSP/3) = true;
in_water = false(1, nSP);
in_water(nSP/3+1:2*nSP/3) = true;
in_CSminus = false(1, nSP);
in_CSminus(2*nSP/3+1:end) = true;

in_Stim = {in_juice, in_water, in_CSminus};
nStim = length(in_Stim);

rgnAMY = strcmp(rgns(:,1),'AMY');
rgnSC = strcmp(rgns(:,1),'SC');
rgnVS = strcmp(rgns(:,1),'VS');
nRegions = size(rgns,1);

nAMY = numel(rgns{rgnAMY,2});
nSC = numel(rgns{rgnSC,2});
nVS = numel(rgns{rgnVS,2});
nUnits = [nAMY, nSC, nVS];

in_AMY = false(sum(nUnits),1);
in_AMY(1:nAMY) = true; 
in_SC = false(sum(nUnits),1);
in_SC(nAMY+1:nAMY+nSC) = true; 
in_VS = false(sum(nUnits),1);
in_VS(nAMY+nSC+1:end) = true; 

in_Rgn = {in_AMY,in_SC,in_VS};
stimNames = {'juice', 'water', 'CS-'};

hist_colors = [0.1 0.2 0.35; 0.75 0.51 0.22; 0.46 0.13 0.13]; % AMY SC VS

%% semi-convert to matt

td = struct;

for iStim = 1:nStim
    during_stim = in_Stim{iStim};
    td(iStim).type = stimNames{iStim};
    td(iStim).AMY_spikes = D(in_AMY, during_stim)';
    td(iStim).SC_spikes = D(in_SC, during_stim)';
    td(iStim).VS_spikes = D(in_VS, during_stim)'; 
end

% get currents (normalized as in softNormalize as far as I can tell...)


for iTarget = 1:nRegions
    in_target = in_Rgn{iTarget};
    
    for iSource = 1:nRegions
        in_source = in_Rgn{iSource};
        
        % compute all currents
        P_both = J(in_target, in_source) * R_ds(in_source, :);
        
        % compute inhibitory currents
        J_inh = J;
        J_inh(J_inh > 0) = 0;
        P_inh = J_inh(in_target, in_source) * R_ds(in_source, :);
        
        % compute excitatory currents
        J_exc = J;
        J_exc(J_exc < 0) = 0;
        P_exc = J_exc(in_target, in_source) * R_ds(in_source, :);
                
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
        [w, scores, eigen,~,~,mu] = pca(P_both(:,:)', 'Algorithm','svd', 'Centered', center_data, 'Economy',0);
        
        
        for iStim = 1:nStim
            during_stim = in_Stim{iStim};
            td(iStim).(['Curr' rgns{iSource,1}, rgns{iTarget,1}]) = P_both(:,during_stim)';
            
            % combined only: project into low-D space and pick the num_dims requested
            projData = P_both(:,during_stim)'; % # sample points x neurons for a given stim
            temp_proj = (projData - repmat(mu,size(projData,1),1)) * w;
            td(iStim).(['Curr' rgns{iSource,1}, rgns{iTarget,1}, '_pca']) = temp_proj;
            
            % norm of n_PCs requested
            top_3_proj = temp_proj(:, 1:n_PCs);
            norm_proj = zeros(size(top_3_proj,1),1);
            for t = 1:size(top_3_proj,1)
                norm_proj(t) = norm(top_3_proj(t, :));
            end
                        
            td(iStim).(['Curr' rgns{iSource,1}, rgns{iTarget,1}, '_pca_norm']) = norm_proj;
            
            td(iStim).(['Curr' rgns{iSource,1}, rgns{iTarget,1},'_inh']) = P_inh(:,during_stim)';
            td(iStim).(['Curr' rgns{iSource,1}, rgns{iTarget,1},'_exc']) = P_exc(:,during_stim)';
        end
    end
end

% prune bad units by cutting any unit whose overall FR is below threshold
AMY_spikes_all = cell2mat(arrayfun(@(t) td(t).AMY_spikes, 1:nStim, 'UniformOutput', false)');
SC_spikes_all = cell2mat(arrayfun(@(t) td(t).SC_spikes, 1:nStim, 'UniformOutput', false)');
VS_spikes_all = cell2mat(arrayfun(@(t) td(t).VS_spikes, 1:nStim, 'UniformOutput', false)');

% idk our rates are at a different scale
bad_AMY_idx = mean(AMY_spikes_all,1) < 0.0005;
bad_SC_idx = mean(SC_spikes_all,1) < 0.0005;
bad_VS_idx = mean(VS_spikes_all,1) < 0.0005;

for iStim = 1:nStim
    % remove bad data indices
    td(iStim).AMY_spikes = td(iStim).AMY_spikes(:,~bad_AMY_idx);
    td(iStim).SC_spikes = td(iStim).SC_spikes(:,~bad_SC_idx);
    td(iStim).VS_spikes = td(iStim).VS_spikes(:,~bad_VS_idx);
    
    for iSource = 1:nRegions
        % remove target units below FR threshold from currents
        td(iStim).(['Curr' rgns{iSource,1} 'AMY']) = td(iStim).(['Curr' rgns{iSource,1} 'AMY'])(:,~bad_AMY_idx);
        td(iStim).(['Curr' rgns{iSource,1} 'SC']) = td(iStim).(['Curr' rgns{iSource,1} 'SC'])(:,~bad_SC_idx);
        td(iStim).(['Curr' rgns{iSource,1} 'VS']) = td(iStim).(['Curr' rgns{iSource,1} 'VS'])(:,~bad_VS_idx); 
    end
end

% re-count units
nAMY = size(td(iStim).AMY_spikes,2);
nSC = size(td(iStim).SC_spikes,2);
nVS = size(td(iStim).VS_spikes,2);

%% B) Trial-averaged firing rates in the pseudopopulation dataset for
% Monkey D for the amygdala, subcallosal ACC, and striatum during the
% unconditioned stimulus (left, inset number denotes neuron count in each
% region), water stimulus (middle), and juice stimulus (right). Neurons in
% each region are aligned on the presentation time of the stimulus and
% sorted according to their time of peak activity in the juice condition.

% get the sort from the full spikes in juice
a = td(1).AMY_spikes;
[~,idx] = max(a,[],1);
[~,AMY_sort] = sort(idx);
a = td(1).SC_spikes;
[~,idx] = max(a,[],1);
[~,SC_sort] = sort(idx);
a = td(1).VS_spikes;
[~,idx] = max(a,[],1);
[~,VS_sort] = sort(idx);

normFacA = mean(abs(AMY_spikes_all(:,~bad_AMY_idx)),1);
normFacS = mean(abs(SC_spikes_all(:,~bad_SC_idx)),1);
normFacV = mean(abs(VS_spikes_all(:,~bad_VS_idx)),1);

softFac = 0.0005;

% set up figure
figure('color','w');
AxD = arrayfun(@(i) subplot(3,3,i,'NextPlot', 'add', 'Box', 'on', 'TickDir','out', 'FontSize', 12, 'Fontweight', 'bold',  ...
    'CLim', [0 2], 'xtick', '', 'ytick', ''), 1:nRegions^2);
cm = brewermap(100,'*RdGy');

count = 1;
for iStim = 1:nStim
    
    tmpAMY = td(iStim).AMY_spikes ./ (normFacA + softFac);
    tmpSC = td(iStim).SC_spikes ./ (normFacS + softFac);
    tmpVS = td(iStim).VS_spikes ./ (normFacV + softFac);
    
    subplot(nRegions,nStim,count);
    imagesc( tmpAMY(:, AMY_sort)' )
    title('AMY','FontWeight', 'bold'), ylabel([stimNames{iStim}])
    set(gca, 'xcolor', hist_colors(1,:),'ycolor', hist_colors(1,:), 'linewidth', 4)
    
    subplot(nRegions,nStim,count+1);
    imagesc( tmpSC(:, SC_sort)' )
    title('SC','FontWeight', 'bold'), ylabel([stimNames{iStim}])
    set(gca, 'xcolor', hist_colors(2,:),'ycolor', hist_colors(2,:), 'linewidth', 4)
    
    if count==1
        text(gca, 0.5*mean(get(gca,'xlim')), 1.2*max(get(gca,'ylim')), ...
            'target rates', 'fontweight', 'bold', 'fontsize', 13)
    end
    
    subplot(nRegions,nStim,count+2); 
    imagesc( tmpVS(:, VS_sort)' )
    title('VS','FontWeight', 'bold'), ylabel([stimNames{iStim}])
    set(gca, 'xcolor', hist_colors(3,:),'ycolor', hist_colors(3,:), 'linewidth', 4)
    
    count = count + 3;
end

axis(AxD, 'tight'),
set(gcf, 'units', 'normalized', 'outerposition', [0.25 0 0.4 0.8])

oldpos = get(AxD(6),'Position');
colorbar(AxD(6)), colormap(cm);
set(AxD(6),'Position',oldpos)

print('-dtiff', '-r400', [RNNfigdir, 'pseudopop_rates_', RNNname(1:end-4)])
close

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
            plot(P(:,1),'LineWidth',2,'Color', hist_colors(iSource,:))
            
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
plot(fliplr(mean(abs(CurrVStoSC),1)), 'linewidth', 2.5, 'color', hist_colors(3,:))
line(gca,[allEndT(1), allEndT(1)], [min(get(gca,'YLim')) max(get(gca,'YLim'))], 'color', 'black', 'linewidth', 1, 'linestyle', '--')
line(gca,[allEndT(2), allEndT(2)], [min(get(gca,'YLim')) max(get(gca,'YLim'))], 'color', 'black', 'linewidth', 1, 'linestyle', '--')
title('Curr VS > SC')
set(gca,'xtick', allEndT - 0.5*mean(diff(allEndT)) , 'xticklabel', stimNames)

% from ACC to striatum
subplot(2,1,2),
plot(fliplr(mean(abs(CurrSCtoVS),1)), 'linewidth', 2.5, 'color', hist_colors(2,:))
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

subplot(3,3,1), semilogy(histcenters,J_amy_to_amy_N./max(J_amy_to_amy_N), 'o-', 'color', hist_colors(1,:)),  title('AMY to AMY')
subplot(3,3,2), semilogy(histcenters,J_sc_to_amy_N./max(J_sc_to_amy_N), '*-', 'color', hist_colors(1,:)), title('SC to AMY')
subplot(3,3,3), semilogy(histcenters,J_vs_to_amy_N./max(J_vs_to_amy_N), 'v-', 'color', hist_colors(1,:)), title('VS to AMY')

% pjns to SC
J_amy_to_sc = J(in_SC, in_AMY);
J_sc_to_sc = J(in_SC, in_SC);
J_vs_to_sc = J(in_SC, in_VS);

[J_amy_to_sc_N] = histcounts(reshape(J_amy_to_sc(:)./sqrt(nAMY),nAMY*nSC,1),Jlim(1):binwidth:Jlim(2)); % sqrt(nAMY)*
[J_sc_to_sc_N] = histcounts(reshape(J_sc_to_sc(:)./sqrt(nSC),nSC^2,1),Jlim(1):binwidth:Jlim(2)); % sqrt(nSC)*
[J_vs_to_sc_N] = histcounts(reshape(J_vs_to_sc(:)./sqrt(nVS),nVS*nSC,1),Jlim(1):binwidth:Jlim(2)); % sqrt(nVS)*

subplot(3,3,4), semilogy(histcenters,J_amy_to_sc_N./max(J_amy_to_sc_N), 'o-', 'color', hist_colors(2,:)), title('AMY to SC')
subplot(3,3,5), semilogy(histcenters,J_sc_to_sc_N./max(J_sc_to_sc_N), '*-', 'color', hist_colors(2,:)), title('SC to SC')
subplot(3,3,6), semilogy(histcenters,J_vs_to_sc_N./max(J_vs_to_sc_N), 'v-', 'color', hist_colors(2,:)), title('VS to SC')

% pjns to VS

J_amy_to_vs = J(in_VS, in_AMY);
J_sc_to_vs = J(in_VS, in_SC);
J_vs_to_vs = J(in_VS, in_VS);

[J_amy_to_vs_N] = histcounts(reshape(J_amy_to_vs(:)./sqrt(nAMY),nAMY*nVS,1),Jlim(1):binwidth:Jlim(2)); % *sqrt(nAMY)
[J_sc_to_vs_N] = histcounts(reshape(J_sc_to_vs(:)./sqrt(nSC),nSC*nVS,1),Jlim(1):binwidth:Jlim(2)); % sqrt(nSC)*
[J_vs_to_vs_N] = histcounts(reshape(J_vs_to_vs(:)./sqrt(nVS),nVS^2,1),Jlim(1):binwidth:Jlim(2)); % sqrt(nVS)*

subplot(3,3,7), semilogy(histcenters,J_amy_to_vs_N./max(J_amy_to_vs_N), 'o-', 'color', hist_colors(3,:)), title('AMY to VS')
subplot(3,3,8), semilogy(histcenters,J_sc_to_vs_N./max(J_sc_to_vs_N), '*-', 'color', hist_colors(3,:)), title('SC to VS')
subplot(3,3,9), semilogy(histcenters,J_vs_to_vs_N./max(J_vs_to_vs_N), 'v-', 'color', hist_colors(3,:)), title('VS to VS')

arrayfun(@(x) set(AxD(x),'xlim', [-1.25*cmax 1.25*cmax], 'YSCale', 'log', 'ylim', [0.0001 1]), 1:length(AxD));
AxL = findobj(gcf,'Type','Line');
arrayfun(@(x) set(AxL(x), 'linewidth', 1.5, 'markersize', 2), 1:length(AxL));

% TO DO: SET TO PRINT

