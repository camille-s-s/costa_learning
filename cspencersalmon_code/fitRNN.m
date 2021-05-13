function RNN = fitRNN(params)

%% data meta
datadir         = '~/Dropbox (BrAINY Crew)/seqnmf/data/seqnmf analyses/subcal_20201016/K=3_L=1.5/';
monkey          = 'D';
min_win         = 1;
win_event       = 'stimOn';
max_win         = 1500;
min_n_trls      = 20;
smoothwin       = 150;
allStim         = [3 4 5];
dataset         = 'matched_trls';       % choose specific dataset in seqstruct
regionNames     = {'AMY';'SC';'VS'};
nRegions        = length(regionNames);

%% data parameters
dtData          = 0.005;                % time step (in s) of the training data
dtFactor        = 20;                   % number of interpolation steps for RNN
dtRNN           = dtData / dtFactor;    % time step (in s) for integration
normByRegion    = 1;                    % normalize activity by region or globally
stimID          = allStim;              % 3, 4, 5, or [3 4 5]
sortData        = false;
smoothFlag      = false;

%% RNN parameters
g               = 1.5;                  % instability (chaos); g<1=damped, g>1=chaotic
tauRNN          = 0.001;                 % decay costant of RNN units ?in msec
tauWN           = 0.1;                  % decay constant on filtered white noise inputs
ampInWN         = 0.0001;                 % input amplitude of filtered white noise

%% training params
alpha           = 1;                    % overall learning rate for regularizer
nRunTrain       = 1500;
nRunFree        = round(0.01*nRunTrain);
nRunTot         = nRunTrain+nRunFree;   % idk according to CURBD
nonlinearity    = @tanh;                % inline function for nonlinearity
resetPoints     = 1;                    % default to only set initial state at time 1

%% output options
plotStatus      = true;
saveMdl         = true;

%% output directories
RNNdir          = '~/Dropbox (BrAINY Crew)/csRNN/data/';
RNNfigdir       = '~/Dropbox (BrAINY Crew)/csRNN/figures/';
RNNname         = [monkey, '_allrgn_', num2str(min_win), '_', win_event,  '_', num2str(max_win), '_', ...
    num2str(min_n_trls), '_', num2str(smoothwin) '.mat'];

%% RNN

% overwrite defaults based on inputs
if exist('params','var')
    assignParams(who,params);
end

AMYfnm=[monkey, '_' regionNames{1}, '_',  num2str(min_win), '_', win_event, '_', num2str(max_win), '_', ...
    num2str(min_n_trls), '_', num2str(smoothwin), '_stimID_', num2str(allStim), '.mat'];
SCfnm=[monkey,  '_' regionNames{2}, '_', num2str(min_win), '_', win_event, '_', num2str(max_win), '_', ...
    num2str(min_n_trls), '_', num2str(smoothwin), '_stimID_', num2str(allStim), '.mat'];
VSfnm=[monkey, '_' regionNames{3}, '_',  num2str(min_win), '_', win_event, '_', num2str(max_win), '_', ...
    num2str(min_n_trls), '_', num2str(smoothwin), '_stimID_', num2str(allStim), '.mat'];

% load data
load([datadir AMYfnm], 'seqnmf');
AMYData = seqnmf.(dataset)(end).in;
itiAMYData = seqnmf.ITI(end).in;
itiAMYData.train = cell2mat(arrayfun(@(n) interp(itiAMYData.train(n,:),5), 1:size(itiAMYData.train,1), 'UniformOutput', false)');
itiAMYData.trainlabels = repelem(itiAMYData.trainlabels,5);

load([datadir SCfnm], 'seqnmf');
SCData = seqnmf.(dataset)(end).in;
itiSCData = seqnmf.ITI(end).in;
itiSCData.train = cell2mat(arrayfun(@(n) interp(itiSCData.train(n,:),5), 1:size(itiSCData.train,1), 'UniformOutput', false)');
itiSCData.trainlabels = repelem(itiSCData.trainlabels,5);

load([datadir VSfnm], 'seqnmf');
VSData = seqnmf.(dataset)(end).in;
itiVSData = seqnmf.ITI(end).in;
itiVSData.train = cell2mat(arrayfun(@(n) interp(itiVSData.train(n,:),5), 1:size(itiVSData.train,1), 'UniformOutput', false)');
itiVSData.trainlabels = repelem(itiVSData.trainlabels,5);


% data partitioning by condition
dataLabels = [];
itiDataLabels = [];
AMYRates = [];
SCRates = [];
VSRates = [];
AMYitiRates = [];
SCitiRates = [];
VSitiRates = [];
for stim = 1:length(stimID)
    stimnum = stimID(stim);
    dataLabels = [dataLabels stimnum*ones(1,sum(AMYData.trainlabels==stimnum))];
    itiDataLabels = [itiDataLabels stimnum*ones(1,sum(itiAMYData.trainlabels==stimnum))];
    AMYRates = [AMYRates AMYData.train(:,AMYData.trainlabels==stimnum)];
    SCRates = [SCRates SCData.train(:,SCData.trainlabels==stimnum)];
    VSRates = [VSRates VSData.train(:,VSData.trainlabels==stimnum)];
    
    AMYitiRates = [AMYitiRates itiAMYData.train(:,itiAMYData.trainlabels==stimnum)];
    SCitiRates = [SCitiRates itiSCData.train(:,itiSCData.trainlabels==stimnum)];
    VSitiRates = [VSitiRates itiVSData.train(:,itiVSData.trainlabels==stimnum)];
end

ALLRates = [AMYRates;SCRates;VSRates];

% load params from data - overwrite if necessary
if strcmp(dataset, 'raw_matched_trls')
    Fs = 1000;
    Lbins= size(AMYRates,2)/(length(stimID)*0.75*min_n_trls);%seqnmf.params.Lneural;
    dtData = 1 / Fs;
    dtRNN = dtData / dtFactor;
else
    Fs = seqnmf.meta.Fs;
    Lbins = seqnmf.params.Lneural;
end

% if concatenating multi-trial data
resetPoints = [];
for stim = 1:length(stimID)
    resetPoints = [resetPoints (stim*Lbins/Fs)/dtRNN+1];
end

% remove nonfiring units
AMYitiRates(mean(AMYRates,2)==0,:)=[];
SCitiRates(mean(SCRates,2)==0,:)=[];
VSitiRates(mean(VSRates,2)==0,:)=[];

AMYRates(mean(AMYRates,2)==0,:)=[];
SCRates(mean(SCRates,2)==0,:)=[];
VSRates(mean(VSRates,2)==0,:)=[];
ALLRates(mean(ALLRates,2)==0,:)=[];

% finish setting up default parameters
nAMY = size(AMYRates,1);
nSC = size(SCRates,1);
nVS = size(VSRates,1);
nUnits = nAMY + nSC + nVS;
nLearn = nUnits; % number of learning steps

% very coarse sanity checking
if ~isequal(nUnits,size(ALLRates,1))
    keyboard
end

if dtData*Fs ~= 1
    keyboard
end

% smooth - stick ITI in front and trim off for padding TO DO: CHANGE THIS
% TO WORK ON NON TRIAL AVERAGED STUFF (used to be after trial averaging)
if smoothFlag
    
    ALL_smoothed_trl_avg = [];
    
    for stim = 1:length(stimID)
        iStart = (stim-1)*Lbins+1;
        iStop = stim*Lbins;
        
        % remove any edge artifacts
        trace_diff = (ALL_iti_avg(:,iStop-100)-ALL_trl_avg(:,iStart));
        tmp = [ALL_iti_avg(:,iStart:iStop-100) - trace_diff, ALL_trl_avg(:,iStart:iStop)];
        
        % smooth
        tmp = smoothdata(tmp,2,'movmean', 0.1*Fs);
        
        % trim off ITI data
        tmp(:,1:Lbins-100) = [];
        
        ALL_smoothed_trl_avg = [ALL_smoothed_trl_avg, tmp];
        
    end
    
    ALL_trl_avg = ALL_smoothed_trl_avg;
end

% trial average within trial types
AMY_trl_avg = [];
SC_trl_avg = [];
VS_trl_avg = [];

AMY_iti_avg = [];
SC_iti_avg = [];
VS_iti_avg = [];

for stim = 1:length(stimID)
    AMYtmp = AMYRates(:,dataLabels==stimID(stim));
    SCtmp = SCRates(:,dataLabels==stimID(stim));
    VStmp = VSRates(:,dataLabels==stimID(stim));
    
    AMY_trl_avg = [AMY_trl_avg, cell2mat(arrayfun(@(n) mean(reshape(AMYtmp(n,:),Lbins,size(AMYtmp,2)/Lbins),2)', ...
        1:nAMY,'uniformoutput',false)')]; 
    SC_trl_avg = [SC_trl_avg, cell2mat(arrayfun(@(n) mean(reshape(SCtmp(n,:),Lbins,size(SCtmp,2)/Lbins),2)', ...
        1:nSC,'uniformoutput',false)')]; 
    VS_trl_avg = [VS_trl_avg, cell2mat(arrayfun(@(n) mean(reshape(VStmp(n,:),Lbins,size(VStmp,2)/Lbins),2)', ...
        1:nVS,'uniformoutput',false)')]; 
    
    
    AMYitiTmp = AMYitiRates(:,itiDataLabels==stimID(stim));
    SCitiTmp = SCitiRates(:,itiDataLabels==stimID(stim));
    VSitiTmp = VSitiRates(:,itiDataLabels==stimID(stim));
    
    AMY_iti_avg = [AMY_iti_avg, cell2mat(arrayfun(@(n) mean(reshape(AMYitiTmp(n,:),Lbins,size(AMYitiTmp,2)/Lbins),2)', ...
        1:nAMY,'uniformoutput',false)')]; 
    SC_iti_avg = [SC_iti_avg, cell2mat(arrayfun(@(n) mean(reshape(SCitiTmp(n,:),Lbins,size(SCitiTmp,2)/Lbins),2)', ...
        1:nSC,'uniformoutput',false)')]; 
    VS_iti_avg = [VS_iti_avg, cell2mat(arrayfun(@(n) mean(reshape(VSitiTmp(n,:),Lbins,size(VSitiTmp,2)/Lbins),2)', ...
        1:nVS,'uniformoutput',false)')];
end

ALL_iti_avg = [AMY_iti_avg; SC_iti_avg; VS_iti_avg];
ALL_trl_avg = [AMY_trl_avg; SC_trl_avg; VS_trl_avg];



% sort by order of max val for ease of visualization
if sortData
    AMY_max = cell2mat(arrayfun(@(n) find(AMY_trl_avg(n,:)==max(AMY_trl_avg(n,:),[],2),1), 1:nAMY, 'UniformOutput', false))';
    SC_max = cell2mat(arrayfun(@(n) find(SC_trl_avg(n,:)==max(SC_trl_avg(n,:),[],2),1), 1:nSC, 'UniformOutput', false))';
    VS_max = cell2mat(arrayfun(@(n) find(VS_trl_avg(n,:)==max(VS_trl_avg(n,:),[],2),1), 1:nVS, 'UniformOutput', false))';
    ALL_max = cell2mat(arrayfun(@(n) find(ALL_trl_avg(n,:)==max(ALL_trl_avg(n,:),[],2),1), 1:nUnits, 'UniformOutput', false))';
    
    [~, AMY_sort] = sort(AMY_max);
    [~, SC_sort] = sort(SC_max);
    [~, VS_sort] = sort(VS_max);
    [~, max_sort] = sort(ALL_max);
    
    AMY_trl_avg = AMY_trl_avg(AMY_sort,:);
    SC_trl_avg = SC_trl_avg(SC_sort,:);
    VS_trl_avg = VS_trl_avg(VS_sort,:);
    ALL_trl_avg = ALL_trl_avg(max_sort,:);
end

% divid by mean and scale to one by region or globally - TO DO: SWITCH TO
% JUST DIVIDING BY MAX OF EACH REGION
if normByRegion
    AMYnorm = AMY_trl_avg; %AMY_trl_avg ./ mean(AMY_trl_avg,2);
    SCnorm = SC_trl_avg; %SC_trl_avg ./ mean(SC_trl_avg,2);
    VSnorm = VS_trl_avg; %VS_trl_avg ./ mean(VS_trl_avg,2);
    
    % divide each region by its mean-normed max
    AMYnorm = AMYnorm./max(AMYnorm(:));
    SCnorm = SCnorm./max(SCnorm(:));
    VSnorm = VSnorm./max(VSnorm(:));
    
    % concatenate regions
    nonnormed_rates = [AMY_trl_avg;SC_trl_avg;VS_trl_avg];
    target_rates = [AMYnorm;SCnorm;VSnorm];
else
    ALLnorm = ALL_trl_avg;%ALL_trl_avg ./ mean(ALL_trl_avg,2);
    ALLnorm = ALLnorm./max(ALLnorm(:));
    
    nonnormed_rates = ALL_trl_avg;
    target_rates = ALLnorm;
end

target_rates = min(target_rates, 0.999);
target_rates = max(target_rates, -0.999);

% set up indexing vectors for submatrices
in_AMY = false(nUnits,1);
in_AMY(1:nAMY) = true;
in_SC = false(nUnits,1);
in_SC(nAMY+1:nAMY+nSC) = true;
in_VS = false(nUnits,1);
in_VS(nAMY+nSC+1:end) = true;

if plotStatus
    figure, subplot(1,2,1), imagesc(nonnormed_rates), title('non-normed rates'), colorbar, subplot(1,2,2), imagesc(target_rates), colorbar, title('target_rates')
    print('-dtiff', '-r400', [RNNfigdir, 'targets_', RNNname(1:end-4)])
    close
end

% housekeeping
if any(isnan(target_rates(:)))
    keyboard
end

% if the RNN is bigger than training neurons, pick the ones to target -
% NOTE: I DON'T QUITE GET THIS
if sortData
    learnList = 1:nUnits;
else
    learnList = randperm(nUnits);
end

iTarget = learnList(1:nLearn);
iNonTarget = learnList(nLearn:end);

% set up data vectors
tData = dtData*(0:size(target_rates,2)-1); % time vector for actual data
tRNN = 0:dtRNN:tData(end); % timevec for RNN
if ~isequal(length(tData),size(target_rates,2))
    keyboard
end

% set up white noise inputs (from CURBD)
ampWN = sqrt(tauWN/dtRNN);
iWN = ampWN*randn(nUnits, length(tRNN));
inputWN = ones(nUnits, length(tRNN));
for tt = 2: length(tRNN)
    inputWN(:, tt) = iWN(:, tt) + (inputWN(:, tt - 1) - iWN(:, tt))*exp(-(dtRNN/tauWN));
end
inputWN = ampInWN*inputWN;

% initialize DI matrix J
J = g * randn(nUnits,nUnits) / sqrt(nUnits);
J0 = J;

% get standard deviation of entire data
stdData = std(reshape(target_rates(iTarget,:), length(iTarget)*length(tData), 1));
% stdData = std(target_rates(:));

% get indices for each sample of model data
iModelSample = zeros(length(tData), 1);
for i=1:length(tData)
    [~, iModelSample(i)] = min(abs(tData(i)-tRNN));
end

% initialize some others
R = zeros(nUnits, length(tRNN)); % rate matrix - firing rates of neurons
chi2 = zeros(1,nRunTot);
pVars = zeros(1,nRunTot);
JR = zeros(nUnits, length(tRNN)); % z(t) for the output readout unit

% initialize learning update matrix (see Sussillo and Abbot, 2009)
PJ = alpha*eye(nUnits); % dim are pN x pN where p=fraction of neurons to modify - here it's all of them

% initialize outstruct fieldnames
if isequal(stimID,3)
    stimType = 'juice';
elseif isequal(stimID,4)
    stimType = 'water';
elseif isequal(stimID,5)
    stimType = 'CSminus';
else
    stimType = 'all';
end

% convert labels into a regions list
%   cell array, column 1 is region name, column 2 is indices in RNN

regionIDs = {1:nAMY; ...
    nAMY+1:nAMY+nSC; ...
    nAMY+nSC+1:nUnits};
% now put it all together
regions = cell(nRegions,2);
regions(:,1) = regionNames;
regions(:,2) = regionIDs;


if plotStatus
    f = figure('Position',[100 100 1800 600]);
end
%% training
% loop through training runs
for nRun=1:nRunTot
    
    % set initial condition to match target data
    H=target_rates(:,1);
    
    % convert to currents through nonlinearity
    R(:, 1) = nonlinearity(H);
    
    tLearn=0; % keeps track of current time
    iLearn=1; % keeps track of last data point learned
    
    for tt = 2:length(tRNN) % why start from 2?
        tLearn = tLearn + dtRNN;
        
        % check if the current index is a reset point. Typically this won't
        % be used, but it's an option for concatenating multi-trial data
        if ismember(tt,resetPoints)
            H = target_rates(:,floor(tt/dtFactor)+1);
        end
        
        % compute next RNN step
        R(:, tt)= nonlinearity(H); %1./(1+exp(-H));
        JR(:, tt) = J*R(:, tt) + inputWN(:,tt);  % zi(t)=sum (Jij rj) over j
        
        H = H + dtRNN * (-H + JR(:,tt))/tauRNN; % update activity
        
        % check if the RNN time coincides with a data point to update J
        if tLearn>=dtData
            tLearn=0;
            
            % error signal --> z(t)-f(t), where f(t) = target function
            % if targets are treated as currents, compare JR
            % if targets treated as smoothed rates, compare RNN
            error = R(1:nUnits, tt) - target_rates(1:nUnits, iLearn);
            
            % update chi2 using this error
            chi2(nRun) = chi2(nRun) + mean(error.^2);
            
            % update learning index
            iLearn=iLearn+1;
            if  (nRun<=nRunTrain)
                
                % update terms for training runs
                k = PJ * R(iTarget, tt); % N x 1
                rPr = R(iTarget, tt)' * k; % scalar; inverse cross correlation of network firing rates
                c=1/(1+rPr); % learning rate
                PJ = PJ - c * (k * k');
                J(1:nUnits, iTarget) = J(1:nUnits, iTarget) - c*error(1:nUnits, :)*k';
            end
            
        end
    end
    
    rModelSample = R(iTarget, iModelSample);
    
    % compute variance explained of activity by units
    pVar = 1 - (norm(target_rates(iTarget,:) - rModelSample, 'fro')/(sqrt(length(iTarget)*length(tData))*stdData)).^2;
    pVars(nRun) = pVar;
    
    % plot
    if plotStatus
        clf(f);
        idx = randi(nUnits);
        subplot(2,4,1);
        hold on;
        imagesc(target_rates(iTarget,:)); colorbar;
        axis tight; set(gca, 'clim', [0 1])
        title('real');
        set(gca,'Box','off','TickDir','out','FontSize',14);
        subplot(2,4,2);
        hold on;
        imagesc(R(iTarget,:)); colorbar;
        axis tight; set(gca, 'clim', [0 1])
        title('model');
        set(gca,'Box','off','TickDir','out','FontSize',14);
        subplot(2,4,[3 4 7 8]);
        hold all;
        plot(tRNN,R(iTarget(idx),:), 'linewidth', 1.5);
        plot(tData,target_rates(iTarget(idx),:), 'linewidth', 1.5);
        ylabel('activity');
        xlabel('time (s)'),
        legend('model','real','location','eastoutside')
        title(nRun)
        set(gca,'Box','off','TickDir','out','FontSize',14);
        subplot(2,4,5);
        hold on;
        plot(pVars(1:nRun));
        ylabel('pVar');
        set(gca,'Box','off','TickDir','out','FontSize',14);
        subplot(2,4,6);
        hold on;
        plot(chi2(1:nRun))
        ylabel('chi2');
        set(gca,'Box','off','TickDir','out','FontSize',14);
        drawnow;
    end
    
    
end

if plotStatus
    %% histogram of whole J
    
    % normalize by sqrt(# presynaptic units)
    % reshapedJplot = sqrt(nUnits)*reshape(J,nUnits^2,1);
    % reshapedJ0plot = sqrt(nUnits)*reshape(J0,nUnits^2,1);
    
    reshapedJplot = reshape(J,nUnits^2,1);
    reshapedJ0plot = reshape(J0,nUnits^2,1);
    
    maxabsval = round(1.02*max(abs([reshapedJplot;reshapedJ0plot])))./2.5;
    hist_colors = [0 0.6 1; 0 0 0.8; 0 0.6 0.6; 0.4 0 0.6; 0.6 0.2 0.8];
    Jlim = [-maxabsval, maxabsval];
    binwidth = (Jlim(2)-Jlim(1))/100;
    
    [bincounts,edgesnew] = histcounts(reshapedJplot,Jlim(1):binwidth:Jlim(2));
    histcenters = edgesnew(1:end-1) + (diff(edgesnew) ./ 2);
    [bincounts0,edgesnew0] = histcounts(reshapedJ0plot,Jlim(1):binwidth:Jlim(2));
    histcenters0 = edgesnew0(1:end-1) + (diff(edgesnew0) ./ 2);
    
    % normalizing each by area under curve
    figure,
    semilogy(histcenters,bincounts./max(bincounts), 'o-', 'color', hist_colors(1,:), 'linewidth', 1.5, 'markersize', 2)
    hold on,
    semilogy(histcenters0,bincounts0./max(bincounts0), 'o-', 'color', hist_colors(2,:), 'linewidth', 1.5, 'markersize', 2)
    legend({'J','J0'}, 'location', 'northeastoutside')
    
    set(gca, 'fontsize', 10, 'ylim', [0.0001 1], 'xlim', Jlim)
    title('log interaction strengths', 'fontsize', 12, 'fontweight', 'bold')
    print('-dtiff', '-r400', [RNNfigdir, 'Jhists_', RNNname(1:end-4)])
    
    %% imagesc of whole J
    figure,
    Jnorm = J; %sqrt(nUnits)*
    J0norm = J0; %sqrt(nUnits)*
    
    cmax = max(abs(([Jnorm(:);J0norm(:)])));
    subplot(1,2,2), imagesc(Jnorm, 'CDataMapping', 'scaled'), title('Jnorm', 'fontsize', 12, 'fontweight','bold'),
    colorbar, colormap(jet), axis square
    xlabel('pre-synaptic'), ylabel('postsynaptic'),
    set(gca, 'xticklabel', '', 'yticklabel', '', 'fontsize', 10, 'fontweight', 'bold', 'clim', [-cmax cmax])
    
    subplot(1,2,1), imagesc(J0norm, 'CDataMapping', 'scaled'), title('J0norm', 'fontsize', 12, 'fontweight','bold'),
    colorbar, colormap(jet), axis square
    xlabel('pre-synaptic'), ylabel('postsynaptic'),
    set(gca,  'xticklabel', '', 'yticklabel', '', 'fontsize', 10, 'fontweight', 'bold', 'clim', [-cmax cmax])
    print('-dtiff', '-r400', [RNNfigdir, 'J_', RNNname(1:end-4)])
    
    %% histograms of subJs
    figure('color','w');
    AxH = arrayfun(@(i) subplot(3,3,i,'NextPlot', 'add', 'Box', 'off'), 1:nRegions^2);
    % hold on,
    
    % pjns to AMY
    J_amy_to_amy = J(in_AMY,in_AMY);
    J_sc_to_amy = J(in_AMY, in_SC);
    J_vs_to_amy = J(in_AMY, in_VS);
    
    [J_amy_to_amy_N] = histcounts(reshape(J_amy_to_amy,nAMY^2,1),Jlim(1):binwidth:Jlim(2)); % sqrt(nAMY)*
    [J_sc_to_amy_N] = histcounts(reshape(J_sc_to_amy,nSC*nAMY,1),Jlim(1):binwidth:Jlim(2)); % sqrt(nSC)*
    [J_vs_to_amy_N] = histcounts(reshape(J_vs_to_amy,nVS*nAMY,1),Jlim(1):binwidth:Jlim(2)); % sqrt(nVS)*
    
    subplot(3,3,1), semilogy(histcenters,J_amy_to_amy_N./max(J_amy_to_amy_N), 'o-', 'color', hist_colors(1,:)),  title('AMY to AMY')
    subplot(3,3,2), semilogy(histcenters,J_sc_to_amy_N./max(J_sc_to_amy_N), '*-', 'color', hist_colors(1,:)), title('SC to AMY')
    subplot(3,3,3), semilogy(histcenters,J_vs_to_amy_N./max(J_vs_to_amy_N), 'v-', 'color', hist_colors(1,:)), title('VS to AMY')
    
    % pjns to SC
    J_amy_to_sc = J(in_SC, in_AMY);
    J_sc_to_sc = J(in_SC, in_SC);
    J_vs_to_sc = J(in_SC, in_VS);
    
    [J_amy_to_sc_N] = histcounts(reshape(J_amy_to_sc,nAMY*nSC,1),Jlim(1):binwidth:Jlim(2)); % sqrt(nAMY)*
    [J_sc_to_sc_N] = histcounts(reshape(J_sc_to_sc,nSC^2,1),Jlim(1):binwidth:Jlim(2)); % sqrt(nSC)*
    [J_vs_to_sc_N] = histcounts(reshape(J_vs_to_sc,nVS*nSC,1),Jlim(1):binwidth:Jlim(2)); % sqrt(nVS)*
    
    subplot(3,3,4), semilogy(histcenters,J_amy_to_sc_N./max(J_amy_to_sc_N), 'o-', 'color', hist_colors(2,:)), title('AMY to SC')
    subplot(3,3,5), semilogy(histcenters,J_sc_to_sc_N./max(J_sc_to_sc_N), '*-', 'color', hist_colors(2,:)), title('SC to SC')
    subplot(3,3,6), semilogy(histcenters,J_vs_to_sc_N./max(J_vs_to_sc_N), 'v-', 'color', hist_colors(2,:)), title('VS to SC')
    
    % pjns to VS
    
    J_amy_to_vs = J(in_VS, in_AMY);
    J_sc_to_vs = J(in_VS, in_SC);
    J_vs_to_vs = J(in_VS, in_VS);
    
    [J_amy_to_vs_N] = histcounts(reshape(J_amy_to_vs,nAMY*nVS,1),Jlim(1):binwidth:Jlim(2)); % *sqrt(nAMY)
    [J_sc_to_vs_N] = histcounts(reshape(J_sc_to_vs,nSC*nVS,1),Jlim(1):binwidth:Jlim(2)); % sqrt(nSC)*
    [J_vs_to_vs_N] = histcounts(reshape(J_vs_to_vs,nVS^2,1),Jlim(1):binwidth:Jlim(2)); % sqrt(nVS)*
    
    subplot(3,3,7), semilogy(histcenters,J_amy_to_vs_N./max(J_amy_to_vs_N), 'o-', 'color', hist_colors(3,:)), title('AMY to VS')
    subplot(3,3,8), semilogy(histcenters,J_sc_to_vs_N./max(J_sc_to_vs_N), '*-', 'color', hist_colors(3,:)), title('SC to VS')
    subplot(3,3,9), semilogy(histcenters,J_vs_to_vs_N./max(J_vs_to_vs_N), 'v-', 'color', hist_colors(3,:)), title('VS to VS')
    
    arrayfun(@(x) set(AxH(x),'xlim', [-cmax/2.5 cmax/2.5], 'YSCale', 'log', 'ylim', [0.0001 1]), 1:length(AxH));
    AxL = findobj(gcf,'Type','Line');
    arrayfun(@(x) set(AxL(x), 'linewidth', 1.5, 'markersize', 2), 1:length(AxL));
    
    
    print('-dtiff', '-r400', [RNNfigdir, 'Jsubmathists_', RNNname(1:end-4)])
    close
end
%% package up outputs

if  isfile([RNNdir, RNNname])
    load([RNNdir, RNNname],'phrRNN')
else
    RNN = struct;
end

if ~isfield(RNN,'meta')
    RNN.meta.createdwhen = datetime;
    RNN.meta.stimID = allStim;
    RNN.meta.Fs = Fs;
    RNN.meta.inputfile = {[datadir AMYfnm];[datadir SCfnm];[datadir VSfnm]};
    RNN.meta.monkey = monkey;
    RNN.meta.min_win = min_win;
    RNN.meta.max_win = max_win;
    RNN.meta.win_event = win_event;
    RNN.meta.min_n_trls = min_n_trls;
    RNN.meta.smoothwin = smoothwin;
    RNN.meta.dataset = dataset;
end

if ~isfield(RNN,stimType)
    r=1;
else
    r=length(RNN.(stimType))+1; % append repeat runs to struct
end

RNN.(stimType)(r).params = struct( ...
    'smoothFlag', smoothFlag, ...
    'dtFactor',dtFactor, ...
    'normByRegion',normByRegion, ...
    'g',g, ...
    'alpha',alpha, ...
    'tauRNN',tauRNN, ...
    'tauWN',tauWN, ...
    'ampInWN',ampInWN, ...
    'nRunTot',nRunTot, ...
    'nRunTrain',nRunTrain, ...
    'nRunFree',nRunFree, ...
    'nonlinearity',nonlinearity, ...
    'resetPoints',resetPoints, ...
    'nUnits',nUnits, ...
    'nRegions',nRegions, ...
    'Lneural', Lbins, ...
    'Lsec', Lbins/Fs, ...
    'sortData',sortData);
RNN.(stimType)(r).mdl = struct( ...
    'regions',{regions}, ...
    'RNN',R, ...
    'tRNN',tRNN, ...
    'dtRNN',dtRNN, ...
    'target_rates',target_rates, ...
    'nonnormed_rates', nonnormed_rates, ...
    'tData',tData, ...
    'dtData',dtData, ...
    'J',J, ...
    'J0',J0, ...
    'chi2',chi2, ...
    'pVar',pVar, ...
    'stdData',stdData, ...
    'inputWN',inputWN, ...
    'iTarget',iTarget, ...
    'iNonTarget',iNonTarget, ...
    'params',RNN.(stimType)(r).params );

if saveMdl
    save([RNNdir, RNNname],'phrRNN', '-v7.3')
end

end






