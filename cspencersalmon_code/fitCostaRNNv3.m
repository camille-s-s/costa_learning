function RNN = fitCostaRNNv3(RNNname, ...
    allSpikes, allPossTS, allEvents, arrayUnit, T, arrayRgns, params)
% Largely identical to fitCostaRNN except pre-processing steps.
% 
% OLD VERSION 1. smooth 2. soft normalize 3. scale to [0 1] 4. remove
% outliers
%
% THIS VERSION 1. smooth 2. remove outliers 3. mean subtract for each
% neuron (maybe) 4. soft normalize

rng(42)

%% data meta
% datadir         = '~/Dropbox (BrAINY Crew)/costa_learning/reformatted_data/';
mouseVer        = '';

%% data parameters
dtData          = 0.010;                % time step (in s) of the training data
dtFactor        = 20;                   % number of interpolation steps for RNN
doSmooth        = true;
smoothWidth     = 0.15;                 % in seconds, width of gaussian kernel if doSmooth == true
rmvOutliers     = true;
meanSubtract    = true;
doSoftNorm      = true;
normByRegion    = false;                % normalize activity by region or globally

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
nSamples        = 21;                   % #samples to save from consecutive fitted Js
minLen          = NaN;                  % minimum trial length (ensures equivalent elapsed time across subsampled consecutive fitted Js)

%% RNN

% overwrite defaults based on inputs
if exist('params','var')
    assignParams(who, params);
end

% define output directories
rnnDir          = ['~/Dropbox (BrAINY Crew)/costa_learning/models/', mouseVer, filesep];
rnnSubDir       = [rnnDir, RNNname(strfind(RNNname, '_') + 1 : end), filesep];
rnnFigDir       = ['~/Dropbox (BrAINY Crew)/costa_learning/figures/', mouseVer, filesep];

if ~isfolder(rnnSubDir)
    mkdir(rnnSubDir)
end

if ~isfolder([rnnSubDir, 'WN/'])
    mkdir([rnnSubDir, 'WN/'])
end

if ~isfolder(rnnFigDir)
    mkdir(rnnFigDir)
end

% separate subdirectory for targets
if ~isfolder([rnnDir 'targets' filesep])
    mkdir([rnnDir 'targets' filesep])
end

% set up final params
dtRNN           = dtData / dtFactor;    % time step (in s) for integration
ampWN           = sqrt( tauWN / dtRNN );
nRunFree        = ceil(0.01 * nRunTrain);
nRunTot         = nRunTrain + nRunFree;   % idk according to CURBD

%% preprocess targets by smoothing, normalizing, re-scaling, and outlier removing
   
targets = allSpikes;

% cleaning: smooth with gaussian
if doSmooth
    targets = smoothdata(targets, 2, 'gaussian', smoothWidth / dtData); % convert smoothing kernel from msec to #bins);
end

% cleaning: outlier removal
if rmvOutliers
    figure('color','w');
    set(gcf, 'units', 'normalized', 'outerposition', [0.05 0.1 0.9 0.6])
    AxD = arrayfun(@(i) subplot(1,2,i,'NextPlot', 'add', 'Box', 'on', ...
        'TickDir','out', 'FontSize', 10, 'fontweight', 'bold'), 1:2);
    
    subplot(1, 2, 1), histogram(mean(targets, 2)), title('mean target FRs all units w/ outliers')
    outliers = isoutlier(mean(targets, 2), 'percentiles', [1 99]);
    targets = targets(~outliers, :);
    arrayUnit = arrayUnit(~outliers, :);
    subplot(1, 2, 2), histogram(mean(targets, 2)), title(['mean target FRs all units minus ', num2str(sum(outliers)), ' outliers'])
    
    oldxlim = cell2mat(get(AxD, 'xlim'));
    newxmin = min(oldxlim(:)); newxmax = max(oldxlim(:));
    arrayfun(@(i) set(AxD(i), 'xlim', [newxmin newxmax]), 1:2)
    tmpFigName = RNNname;
    tmpFigName(strfind(tmpFigName, '_')) = ' ';
    text(AxD(2), -0.3 * (newxmax - newxmin), 1.05 * max(get(AxD(2), 'ylim')), tmpFigName, 'fontweight','bold', 'fontsize', 13)
    print('-dtiff', '-r400', [rnnFigDir, 'targets_outlier_comparison_', RNNname])
    close
    
    % update indexing vectors
    for iRgn = 1 : size(arrayRgns, 1)
        arrayRgns{iRgn, 3}(outliers) = [];
    end
    
end

% transformation: center each neuron by subtracting its mean activity
if meanSubtract
    meanTarg = targets - mean(targets, 2);
    targets = meanTarg;
end

% transformation: this will soft normalize a la Churchland papers (so all
% activity is on roughly similar scale)
if doSoftNorm
    normfac = range(targets, 2); % + (dtData * 10); % normalization factor = firing rate range + alpha
    targets = targets ./ normfac;
end

% housekeeping
if any(isnan(targets(:)))
    keyboard
end

%% set up for model

% nSubset = 175;

try % choose targ subset and starting trial
    prevMdls = dir([rnnSubDir, RNNname, '_set*_trial*.mat']);
    allTrialIDs = unique(arrayfun(@(i) ...
        str2double(prevMdls(i).name(strfind(prevMdls(i).name,'trial') + 5 : end - 4)), ...
        1 : length(prevMdls)));
    prevMdl = load([rnnSubDir, dir([rnnSubDir, RNNname, ...
        '_set*_trial', num2str(max(allTrialIDs)), '.mat']).name]);
    prevJ = prevMdl.RNN.mdl.J;
    prevTrl = prevMdl.RNN.mdl.iTrl;
    if trainFromPrev
        disp('trainFromPrev = True. Training from last completed trial...')
        startTrl = prevMdl.RNN.mdl.iTrl + 1; % max(allTrialIDs) + 1;
    else
        disp('trainFromPrev = False. Training from trial 1...')
        startTrl = 1;
        prevTrl = 0;
        % targSubset = randperm(size(targets, 1), nSubset);
    end
    
    clearvars prevMdl
catch
    disp('no last completed trl in dir; starting from trl 1...')
    startTrl = 1;
    prevTrl = 'no last completed trl in dir';
    % targSubset = randperm(size(targets, 1), nSubset);
end

clearvars prevMdls

% for dev, train on subset of available units until model finalized
% targets = targets(targSubset, :);

if plotStatus
    figure, subplot(1,2,1), imagesc(allSpikes), title('non-normed rates'), colorbar, colormap jet, subplot(1,2,2), imagesc(targets), colorbar, title('target rates')
    if ~isfolder([rnnFigDir, 'targets_', RNNname])
        mkdir([rnnFigDir, 'targets_', RNNname])
    end
    print('-dtiff', '-r400', [rnnFigDir, 'targets_', RNNname])
    close
end

clearvars allSpikes

% set up for training
nUnits = size(targets, 1);
nLearn = nUnits; % number of learning steps

% if the RNN is bigger than training neurons, pick the ones to target (??)
learnList = 1 : nLearn;% randperm(nUnits);
iTarget = learnList(1:nLearn);
iNonTarget = learnList(nLearn:end);

% sanity check match in #trials
assert(isequal(sum(allEvents == 1), height(T)))
nTrls = height(T);

% pull trial starts
fixOnInds = [find(allEvents == 1), size(targets, 2)];

if isnan(minLen)
    minLen = min(diff(fixOnInds)); % shortest trial (from fixation to next fixation)
end

% pull event labels by trial (in time relative to start of each trial)
% (1 = fixation, 2 = stim, 3 = choice, 4 =  outcome, 5 = time of next trl fixation)
stimTimeInds = find(allEvents == 2) - find(allEvents == 1);

clearvars allEvents

% get block/set structure (s sets of j trials each)
nTrlsPerSet = diff([find(T.trls_since_nov_stim == 0); height(T) + 1]); % 2022/03/16 edit
nSets = sum(T.trls_since_nov_stim == 0); % 2022/03/16 edit
setID = repelem(1:nSets, nTrlsPerSet)'; % 2022/03/16 edit
clearvars T

% initialize outputs
stdData = zeros(1,nTrls);
JTrls = NaN(nUnits, nUnits, nTrls);

% save variables for generating each trial's currTargets, tData, and tRNN
save([rnnDir 'targets' filesep, RNNname, '_targets'],  'targets', 'fixOnInds', 'allPossTS', 'nTrls', '-v7.3')

for iTrl = startTrl : nTrls % - 1 or nSets - 1
    explodingGradWarn = false; clear fittedConsJ
    fprintf('\n')
    disp([RNNname, ': training trial # ', num2str(iTrl), '.'])
    tic
    
    iStart = fixOnInds(iTrl); % start of trial
    iStop = fixOnInds(iTrl + 1) - 1; % right before start of next trial
    
    % these three variables will be saved in full for each file instead of
    % each snippet for each trial being saved along with the model so they
    % can be loaded separately in post to make training faster
    currTargets = targets(:, iStart:iStop);
    tData = allPossTS(iStart:iStop); % timeVec for current data
    tRNN = tData(1) : dtRNN : tData(end); % timevec for RNN
    
    % get fixed sample time points for subsampling J's and Rs
    sampleTimePoints = round(linspace(stimTimeInds(iTrl), stimTimeInds(iTrl) + minLen, nSamples));

    % set up white noise inputs (from CURBD)
    iWN = ampWN * randn( nUnits, length(tRNN) );
    inputWN = ones(nUnits, length(tRNN));
    
    for tt = 2 : length(tRNN)
        inputWN(:, tt) = iWN(:, tt) + (inputWN(:, tt - 1) - iWN(:, tt)) * exp( -(dtRNN / tauWN) );
    end
    
    inputWN = ampInWN * inputWN;
    
    % sidebar to save inputWN separately in the hopes of making loading
    % RNNs faster
    inputWNDataFnm = [rnnSubDir, 'WN/', RNNname, '_inputWN_trl', num2str(iTrl), '.mat'];
    
    if saveMdl
        save(inputWNDataFnm, 'inputWN', '-v7.3')
    end
    
    % initialize DI matrix J
    if trainFromPrev && iTrl == startTrl && exist('prevJ', 'var')
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
    
    %% training
    
    JLearn = NaN(nUnits, nUnits, size(currTargets, 2));

    % loop through training runs
    for nRun = 1 : nRunTot
        
        % set initial condition to match target data
        H = currTargets(:, 1);
        
        % convert to currents through nonlinearity
        R(:, 1) = nonlinearity(H);
        
        tLearn = 0; % keeps track of current time
        iLearn = 1; % keeps track of last data point learned 

        for tt = 2 : length(tRNN) % why start from 2?
            
            % index in actual RNN time
            tLearn = tLearn + dtRNN;

            % compute next RNN step
            R(:, tt) = nonlinearity(H);
            JR(:, tt) = J * R(:, tt) + inputWN(:, tt);
            
            % update activity
            H = H + dtRNN * (-H + JR(:, tt)) / tauRNN;
            
            % update J if the RNN time coincides with a data point
            if tLearn >= dtData
                tLearn = 0;
                
                % error signal --> z(t)-f(t), where f(t) = target function
                % if currTargets are  currents, compare JR
                % if currTargets are rates, compare RNN
                error = R(1:nUnits, tt) - currTargets(1:nUnits, iLearn);
                
                if norm(error) > 15 % potential exploding gradient problem?
                    disp(['potential exploding gradient problem at data timepoint ' num2str(iLearn), ...
                        ', nRun= ', num2str(nRun), ' trl= ', num2str(iTrl), '. . .'])
                end
                
                % update chi2 using this error
                chi2(nRun) = chi2(nRun) + mean(error.^2);
                
                % update learning index
                iLearn = iLearn+1;
                                
                if (nRun <= nRunTrain)
                    
                    % update term: sq mdl activity
                    k = PJ * R(iTarget, tt); % N x 1
                    
                    % scalar; inv xcorr of ntwk firing rates use xcorr bc
                    % when you square something it magnifies the
                    % sensitivity to changes
                    rPr = R(iTarget, tt)' * k;
                    
                    % learning rate: if big changes in FRs and wanna move
                    % quickly/take big steps (aka momentum) - as closer,
                    % don't wanna overshoot (convergence would take longer)
                    % so take smaller steps
                    c = 1 / (1 + rPr); % tune learning rate by looking at magnitude of model activity
                                        
                    % use squared firing rates (R * R^T) to update PJ -
                    % maybe momentum effect?
                    PJ = PJ - c * (k * k');
                      
                    %% plain J update
                    % (pre-syn wts adjusted according to post-syn target)
                    J(1 : nUnits, iTarget) = J(1 : nUnits, iTarget) - c * error(1 : nUnits, :) * k';
                                        
                    if nRun == nRunTrain
                        JLearn(:, :, iLearn) = J; % thenwhen nRun == nRunTot, compare JLearn2 and JLearn
                    end
                end
                
            end
        end
        
        % JLearnPrevRun = JLearn; % save current run's fitted JLearn as
        % prev for next one
        rModelSample = R(iTarget, iModelSample);
        pVar = 1 - ( norm(currTargets(iTarget,:) - rModelSample, 'fro' ) / ( sqrt(length(iTarget) * length(tData)) * stdData(iTrl)) ).^2;
        pVars(nRun) = pVar;

        if pVar < 0
            disp('pVar < 0!')
        end
        
        % if final run, save JLearn
        if nRun == nRunTot && ~explodingGradWarn
            fittedConsJ = JLearn(:, :, sampleTimePoints);
        end
        
        % plot
        if plotStatus
            clf(f);
            idx = randi(nUnits);
            
            subplot(2,4,1); hold on;
            imagesc(currTargets(iTarget,:)); colormap(jet), colorbar;
            axis tight; set(gca, 'clim', [0 1], 'Box','off','TickDir', 'out', 'FontSize', 14),
            title('real')
            
            subplot(2,4,2); hold on;
            imagesc(R(iTarget,:)); colormap(jet), colorbar;
            axis tight; set(gca, 'clim', [0 1], 'Box','off','TickDir', 'out', 'FontSize', 14)
            title('model');
            
            subplot(2, 4, [3 4 7 8]); hold all;
            plot(tRNN,R(iTarget(idx),:), 'linewidth', 1.5);
            plot(tData,currTargets(iTarget(idx),:), 'linewidth', 1.5);
            axis tight; set(gca, 'ylim', [-0.1 1], 'Box','off', 'TickDir', 'out', 'FontSize', 14)
            ylabel('activity'); xlabel('time (s)'),
            legend('model', 'real', 'location', 'eastoutside')
            title(['run ', num2str(nRun)])
            
            subplot(2,4,5); hold on;
            plot(pVars(1:nRun)); ylabel('pVar');
            set(gca, 'ylim', [-0.1 1], 'Box', 'off', 'TickDir', 'out', 'FontSize', 14);
            title(['current pVar=', num2str(pVars(nRun), '%.3f')])
            
            subplot(2,4,6); hold on;
            plot(chi2(1:nRun)); ylabel('chi2');
            set(gca, 'ylim', [-0.1 1], 'Box', 'off','TickDir', 'out', 'FontSize', 14);
            title(['current chi2=', num2str(chi2(nRun), '%.3f')])
            drawnow;
        end
                
    end
    
    % get elapsed time
    toc
    
    % save J for next trial
    JTrls(:, :, iTrl) = J;
    
    % package up and save outputs at the end of training for each link
    RNN = struct;
    
    if setID(iTrl) == 1 % untested
        rnnParams = struct( ...
            'iTrl',                 iTrl, ...
            'doSmooth',             doSmooth, ...
            'smoothWidth',          smoothWidth, ...
            'meanSubtract',         meanSubtract, ...
            'doSoftNorm',           doSoftNorm, ...
            'normByRegion',         normByRegion, ...
            'rmvOutliers',          rmvOutliers, ...
            'outliers',             outliers, ...
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
            'arrayUnit',            {arrayUnit}, ...
            'arrayRegions',         {arrayRgns});
    else
        rnnParams = [];
    end
    
    RNN.mdl = struct( ...
        'iTrl',                 iTrl, ...
        'prevTrl',              prevTrl, ...
        'setID',                setID(iTrl), ...
        'RMdlSample',           rModelSample, ...
        'tRNN',                 [], ...
        'dtRNN',                dtRNN, ...
        'targets',              [], ...
        'tData',                [], ...
        'dtData',               dtData, ...
        'J',                    J, ...
        'J0',                   J0, ...
        'fittedConsJ',          fittedConsJ, ...
        'sampleTimePoints',     sampleTimePoints, ...
        'chi2',                 chi2, ...
        'pVars',                pVars, ...
        'stdData',              stdData(iTrl), ...
        'inputWN',              [], ...
        'iTarget',              iTarget, ...
        'iNonTarget',           iNonTarget, ...
        'params',               rnnParams );
    
    if saveMdl
        save([rnnSubDir, RNNname, '_set', num2str(setID(iTrl)), '_trial', num2str(iTrl), '.mat'],'RNN', '-v7.3')
    end
    
    clear RNN fittedConsJ JLearnPrev    
    
end

end






