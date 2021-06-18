function RNN = fitCostaRNN(RNNname, ...
    allSpikes, allPossTS, allEvents, spikeInfo, T, arrayRgns, params)
rng(42)

%% TO DO

% decide what networks stats to save
% decide what other params to save
% decide whether to train on a trial or a set or all trials in a set
% decide what Js to save and what Js to feed in to each link in the chain -
% do you start with a new J for every set or???
% do you re-initialize off the first datapoint for every trial or???
%  DOES IT MATTER THAT WE ARE INITIALIZING DIFFERENT FROZEN WHITE NOISE
% INPUTS EVERY TIME WE START TRAINING A CHUnk?

%% data meta
% datadir         = '~/Dropbox (BrAINY Crew)/costa_learning/reformatted_data/';

%% data parameters
dtData          = 0.020;                % time step (in s) of the training data
dtFactor        = 20;                   % number of interpolation steps for RNN
doSmooth        = true;
doSoftNorm      = true;
normByRegion    = false;                % normalize activity by region or globally
rmvOutliers     = false;

%% RNN parameters
g               = 1.5;                  % instability (chaos); g<1=damped, g>1=chaotic
tauRNN          = 0.001;                % decay costant of RNN units ?in msec
tauWN           = 0.1;                  % decay constant on filtered white noise inputs
ampInWN         = 0.001;                % input amplitude of filtered white noise

%% training params
alpha           = 1;                    % overall learning rate for regularizer
nonlinearity    = @tanh;                % inline function for nonlinearity
nRunTrain       = 1000;
resetPoints     = 1;                    % default to only set initial state at time 1
trainFromPrev   = false;                % assume you're starting from beginning, but if not, feed in previous J

%% output options
plotStatus      = true;
saveMdl         = true;

%% output directories
rnnDir          = '~/Dropbox (BrAINY Crew)/costa_learning/models/';
rnnFigDir       = '~/Dropbox (BrAINY Crew)/costa_learning/models/figures/';

%% RNN

% overwrite defaults based on inputs
if exist('params','var')
    assignParams(who,params);
end

% set up final params
dtRNN           = dtData / dtFactor;    % time step (in s) for integration
ampWN           = sqrt( tauWN / dtRNN );
nRunFree        = ceil(0.01 * nRunTrain);
nRunTot         = nRunTrain + nRunFree;   % idk according to CURBD

% preprocess targets by smoothing, normalizing, re-scaling, and outlier removing
targets = allSpikes;

if doSmooth
    targets = smoothdata(targets, 2, 'movmean', 0.1 / dtData); % convert smoothing kernel from msec to #bins);
end

% this will soft normalize a la Churchland papers
if doSoftNorm
    normfac = range(targets, 2); % + (dtData * 10); % normalization factor = firing rate range + alpha
    targets = targets ./ normfac;
end

if normByRegion
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

if rmvOutliers
    outliers = isoutlier(mean(targets, 2), 'percentiles', [1 99]);
    targets = targets(~outliers, :);
end

% housekeeping
if any(isnan(targets(:)))
    keyboard
end

if plotStatus
    figure, subplot(1,2,1), imagesc(allSpikes), title('non-normed rates'), colorbar, colormap jet, subplot(1,2,2), imagesc(targets), colorbar, title('target rates')
    if ~isfolder([rnnFigDir, 'targets_', RNNname])
        mkdir([rnnFigDir, 'targets_', RNNname])
    end
    print('-dtiff', '-r400', [rnnFigDir, 'targets_', RNNname])
    close
end

% set up for training
nUnits = size(targets, 1);
nLearn = nUnits; % number of learning steps

% if the RNN is bigger than training neurons, pick the ones to target (??)
learnList = randperm(nUnits);
iTarget = learnList(1:nLearn);
iNonTarget = learnList(nLearn:end);

% T = T{1}; % kept both trlInfo but they are almost identical between NIPs

% sanity check match in #trials
assert(isequal(sum(allEvents == 1), height(T)))
nTrls = height(T);

% pull trial starts
fixOnInds = [find(allEvents == 1), size(targets, 2)];
% outcomeInds = find(allEvents == 4);

% get block/set structure (s sets of j trials each)
firstTrlInd = find(T.trls_since_nov_stim == 0);
lastTrlInd = find(T.trls_since_nov_stim  == 0) - 1;

if lastTrlInd(1) ~= 0, keyboard, end
if firstTrlInd(1) ~= 1, keyboard, end
firstTrlInd(1) = [];
lastTrlInd(1:2) = [];
lastTrlInd = [lastTrlInd; height(T)]; % can't have a 0 ind

nTrlsPerSet = T.trls_since_nov_stim(lastTrlInd) + 1; % 10 <= j <= 30 according to paper
nSets = length(nTrlsPerSet); % s <= 32 according to paper
setID = [0; repelem(1:nSets, nTrlsPerSet)'];

% initialize outputs
stdData = zeros(1,nTrls);
JTrls = NaN(nUnits, nUnits, nTrls);


if trainFromPrev
    prevMdls = dir([rnnDir, RNNname, '_set*_trial*.mat']);
    allTrialIDs = unique(arrayfun(@(i) ...
        str2double(prevMdls(i).name(strfind(prevMdls(i).name,'trial') + 5 : end - 4)), ...
        1 : length(prevMdls)));
    startTrl = max(allTrialIDs);
    
    try
        prevMdl = load(prevMdls(find(allTrialIDs == startTrl)).name);
        prevJ = prevMdl.RNN.mdl.J;
        clear prevMdl
    catch
        startTrl = 1;
    end
else
    startTrl = 1;
end



for iTrl = startTrl : nTrls % - 1 or nSets - 1

    fprintf('\n')
    
    disp(['Training trial # ', num2str(iTrl), '.'])
    
    iStart = fixOnInds(iTrl); % start of trial
    iStop = fixOnInds(iTrl + 1) - 1; % right before start of next trial
    currTargets = targets(:, iStart:iStop);
    
    tData = allPossTS(iStart:iStop); % timeVec for current data
    tRNN = tData(1) : dtRNN : tData(end); % timevec for RNN
    
    % set up white noise inputs (from CURBD)
    iWN = ampWN * randn( nUnits, length(tRNN) );
    inputWN = ones(nUnits, length(tRNN));
    for tt = 2: length(tRNN)
        inputWN(:, tt) = iWN(:, tt) + (inputWN(:, tt - 1) - iWN(:, tt)) * exp( -(dtRNN / tauWN) );
    end
    
    inputWN = ampInWN * inputWN;
    
    % initialize DI matrix J
    if trainFromPrev && exist('prevJ', 'var')
        J = prevJ;
    else
        if iTrl == 1
            J = g * randn(nUnits,nUnits) / sqrt(nUnits);
        else
            J = squeeze(JTrls(:, :, iTrl - 1));
        end
    end
    
    J0 = J;
    
    % get standard deviation of entire data that we are looking at
    stdData(iTrl)  = std(reshape(currTargets(iTarget,:), length(iTarget)*length(tData), 1));
    
    % get indices for each sample of model data for getting pVar
    iModelSample = zeros(length(tData), 1);
    for i = 1:length(tData)
        [~, iModelSample(i)] = min(abs(tData(i) - tRNN));
    end
    
    % initialize some others
    R = zeros(nUnits, length(tRNN)); % rate matrix - firing rates of neurons
    chi2 = zeros(1, nRunTot);
    pVars = zeros(1, nRunTot);
    JR = zeros(nUnits, length(tRNN)); % z(t) for the output readout unit
    
    % initialize learning update matrix (see Sussillo and Abbot, 2009)
    PJ = alpha * eye(nUnits); % dim are pN x pN where p=fraction of neurons to modify - here it's all of them
    
    if plotStatus
        f = figure('Position',[100 100 1800 600]);
    end
    
    tic
    
    %% training
    % loop through training runs
    for nRun = 1 : nRunTot
        % set initial condition to match target data
        H = currTargets(:, 1);
        
        % convert to currents through nonlinearity
        R(:, 1) = nonlinearity(H);
        
        tLearn = 0; % keeps track of current time
        iLearn = 1; % keeps track of last data point learned
        
        for tt = 2 : length(tRNN) % why start from 2?
            tLearn = tLearn + dtRNN;
            
            % check if the current index is a reset point. Typically this won't
            % be used, but it's an option for concatenating multi-trial data
            if ismember(tt, resetPoints)
                H = currTargets(:, floor(tt / dtFactor) + 1);
            end
            
            % compute next RNN step
            R(:, tt) = nonlinearity(H); %1./(1+exp(-H));
            JR(:, tt) = J * R(:, tt) + inputWN(:, tt);  % zi(t)=sum (Jij rj) over j
            H = H + dtRNN * (-H + JR(:, tt)) / tauRNN; % update activity
            
            % check if the RNN time coincides with a data point to update J
            if tLearn >= dtData
                tLearn = 0;
                
                % error signal --> z(t)-f(t), where f(t) = target function
                % if currTargets are treated as currents, compare JR
                % if currTargets treated as smoothed rates, compare RNN
                error = R(1:nUnits, tt) - currTargets(1:nUnits, iLearn);
                
                % update chi2 using this error
                chi2(nRun) = chi2(nRun) + mean(error.^2);
                
                % update learning index
                iLearn = iLearn+1;
                if (nRun <= nRunTrain)
                    
                    % update terms for training runs
                    k = PJ * R(iTarget, tt); % N x 1
                    rPr = R(iTarget, tt)' * k; % scalar; inverse cross correlation of network firing rates
                    c = 1 / (1 + rPr); % learning rate
                    PJ = PJ - c * (k * k');
                    J(1:nUnits, iTarget) = J(1:nUnits, iTarget) - c * error(1:nUnits, :) * k';
                end
                
            end
        end
        
        rModelSample = R(iTarget, iModelSample);
        
        % compute variance explained of activity by units
        pVar = 1 - ( norm(currTargets(iTarget,:) - rModelSample, 'fro' ) / ( sqrt(length(iTarget) * length(tData)) * stdData(iTrl)) ).^2;
        pVars(nRun) = pVar;
        
        % plot
        if plotStatus
            clf(f);
            idx = randi(nUnits);
            subplot(2,4,1);
            hold on;
            imagesc(currTargets(iTarget,:)); colormap(jet), colorbar;
            axis tight; set(gca, 'clim', [0 1])
            title('real');
            set(gca,'Box','off','TickDir','out','FontSize',14);
            
            subplot(2,4,2);
            hold on;
            imagesc(R(iTarget,:)); colormap(jet), colorbar;
            axis tight; set(gca, 'clim', [0 1])
            title('model');
            set(gca,'Box','off','TickDir','out','FontSize',14);
            
            subplot(2,4,[3 4 7 8]);
            hold all;
            plot(tRNN,R(iTarget(idx),:), 'linewidth', 1.5);
            plot(tData,currTargets(iTarget(idx),:), 'linewidth', 1.5);
            axis tight; set(gca, 'ylim', [-0.1 1])
            ylabel('activity');
            xlabel('time (s)'),
            legend('model','real','location','eastoutside')
            title(['run ', num2str(nRun)])
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
        
        if nRun == 1
            fprintf(num2str(nRun))
        elseif mod(nRun, 100) == 0
            fprintf('\n')
            fprintf(num2str(nRun))
        else
            fprintf('.')
        end
        
    end
    
    % save J for next link
   JTrls(:, :, iTrl) = J;
    
    % package up and save outputs at the end of training for each link
    RNN = struct;
   
    if setID(iTrl) == 0
        rnnParams = struct( ...
            'iTrl',                 iTrl, ...
            'doSmooth',             doSmooth, ...
            'doSoftNorm',           doSoftNorm, ...
            'normByRegion',         normByRegion, ...
            'rmvOutliers',          rmvOutliers, ...
            'dtFactor',             dtFactor, ...
            'g',                    g, ...
            'alpha',                alpha, ...
            'tauRNN',               tauRNN, ...
            'tauWN',                tauWN, ...
            'ampInWN',              ampInWN, ...
            'nRunTot',              nRunTot, ...
            'nRunTrain',            nRunTrain, ...
            'nRunFree',             nRunFree, ...
            'nonlinearity',         nonlinearity, ...
            'resetPoints',          resetPoints, ... % will need to change
            'nUnits',               nUnits, ...
            'nTrls',                nTrls, ...
            'nSets',                nSets, ...
            'setID',                setID(iTrl), ...
            'arrayUnit',            {spikeInfo.array}, ...
            'arrayRegions',         {arrayRgns});
    else
        rnnParams = [];
    end
    
    RNN.mdl = struct( ...
        'iTrl',                 iTrl, ...
        'setID',                setID(iTrl), ...
        'RMdlSample',           rModelSample, ...
        'tRNN',                 tRNN, ...
        'dtRNN',                dtRNN, ...
        'targets',              currTargets, ...
        'tData',                tData, ...
        'dtData',               dtData, ...
        'J',                    J, ...
        'J0',                   J0, ...
        'chi2',                 chi2, ...
        'pVars',                pVars, ...
        'stdData',              stdData(iTrl), ...
        'inputWN',              inputWN, ...
        'iTarget',              iTarget, ...
        'iNonTarget',           iNonTarget, ...
        'params',               rnnParams );
    
    if saveMdl
        save([rnnDir, RNNname, '_set', num2str(setID(iTrl)), '_trial', num2str(iTrl), '.mat'],'RNN', '-v7.3')
    end
    
    clear RNN
    toc
    
end

%% package up outputs

%
% RMdlSampleTrls = RMdlSampleTrls(1:iTrl, 1);
% chi2Trls = chi2Trls(1:iTrl,:);
% pVarsTrls = pVarsTrls(1:iTrl,:);



end





