function RNN = fitCostaRNNPredict(RNNname, ...
    allSpikes, allPossTS, allEvents, arrayUnit, T, arrayRgns, params)
% Largely identical to fitCostaRNN except pre-processing steps.
%
% OLD VERSION 1. smooth 2. soft normalize 3. scale to [0 1] 4. remove
% outliers
%
% THIS VERSION 1. smooth 2. remove outliers 3. mean subtract for each
% neuron (maybe) 4. soft normalize AND ALSO (8/31/22) TESTING PREDICTION A
% LA KARPATHY

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
nRunTest        = 1;                    % TO DO: IDK HOW MUCH TO TEST?!?!
trainFromPrev   = false;                % assume you're starting from beginning, but if not, feed in previous J
trainRNN        = true;                 % true by default until it's not!
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

if ~isfolder(rnnFigDir)
    mkdir(rnnFigDir)
end

if ~isfolder([rnnDir, 'train_test_lists'])
    mkdir([rnnDir, 'train_test_lists'])
end

% separate subdir for exp_data ( inputs=exp_data(t), targets=exp_data(t+1) )
if ~isfolder([rnnDir 'exp_data' filesep])
    mkdir([rnnDir 'exp_data' filesep])
end

% set up final params
dtRNN           = dtData / dtFactor;    % time step (in s) for integration
ampWN           = sqrt( tauWN / dtRNN );

%% preprocess targets by smoothing, normalizing, re-scaling, and outlier removing, or load previous

if ~isfile([rnnDir 'exp_data' filesep, RNNname, '_exp_data.mat'])
    [exp_data, outliers] = preprocess_data_for_RNN_training(allSpikes, doSmooth, rmvOutliers, meanSubtract, doSoftNorm, smoothWidth, dtData, arrayUnit, arrayRgns);   
    
    % pull trial starts
    fixOnInds = [find(allEvents == 1), size(exp_data, 2)];
    
    % pull event labels by trial (in time relative to start of each trial)
    stimTimeInds = find(allEvents == 2) - find(allEvents == 1); % (1 = fixation, 2 = stim, 3 = choice, 4 =  outcome, 5 = time of next trl fixation)
    
    % sanity check match in #trials
    assert(isequal(sum(allEvents == 1), height(T)))
    nTrls =  height(T); % height(T); % TO DO: CHANGE THIS BACK AFTER DEV!
    
    % save variables for generating each trial's currTargets, tData, and tRNN
    save([rnnDir 'exp_data' filesep, RNNname, '_exp_data.mat'],  'exp_data', 'fixOnInds', 'stimTimeInds', 'nTrls', 'outliers', '-v7.3')
else
    load([rnnDir 'exp_data' filesep, RNNname, '_exp_data.mat'], 'exp_data', 'fixOnInds', 'stimTimeInds', 'nTrls', 'outliers');
end

clearvars allEvents prevMdls allSpikes

% define minimimum trial length so all sampled Js are in same timescale
if isnan(minLen)
    minLen = min(diff(fixOnInds)); % shortest trial (from fixation to next fixation)
end

%%  get block/set structure (s sets of j trials each)

nTrlsPerSet = diff([find(T.trls_since_nov_stim == 0); height(T) + 1]); % 2022/03/16 edit
nSets = sum(T.trls_since_nov_stim == 0); % 2022/03/16 edit
setID = repelem(1 : nSets, nTrlsPerSet)'; % 2022/03/16 edit
clearvars T

%% set up for training

nUnits = size(exp_data, 1); % change targets name to like...exp_data so t+1 can be targets

% if the RNN is bigger than training neurons, pick the ones to target (??)
learnList = 1 : nUnits; % randperm(nUnits);
iTarget = learnList(1 : nUnits);
iNonTarget = learnList(nUnits : end);

%% generate train trial ID list (if trained via multiple fcn calls)

% define train/test split
nTrlsIncluded = 12;
% nTrlsTrain = round(0.75 * nTrlsIncluded);
% nTrlsTest = nTrlsIncluded - nTrlsTrain;
% 
% if ~isfile([rnnDir, 'train_test_lists' filesep, RNNname, '_train_test_list.mat']) % initialize list if starting at beginning
%     subset_trl_IDs = randperm(nTrls, nTrlsIncluded);
%     train_trl_IDs = subset_trl_IDs(1 : nTrlsTrain); % isequal(all_trl_IDs(ismember(all_trl_IDs, train_trl_IDs)), train_trl_IDs)
%     test_trl_IDs = subset_trl_IDs(nTrlsTrain + 1 : end); % all_trl_IDs(~ismember(all_trl_IDs, sort(train_trl_IDs)));
%     save([rnnDir, 'train_test_lists' filesep, RNNname, '_train_test_list.mat'], 'train_trl_IDs', 'test_trl_IDs')
%     start_trl_num = 1;
% else
%     load([rnnDir, 'train_test_lists' filesep, RNNname, '_train_test_list.mat'], 'train_trl_IDs', 'test_trl_IDs')
%     prevMdls = dir([rnnSubDir, RNNname, '_train_trl*_num*.mat']);
%     trl_IDs_in_dir = arrayfun(@(i) ...
%         str2double(prevMdls(i).name(strfind(prevMdls(i).name, 'trl') + 3 : strfind(prevMdls(i).name, '_num') - 1)), ...
%         1 : length(prevMdls));
%     
%     % see which of the trial IDs in the train list are already in the directory
%     last_completed_trl_num = find(~ismember(train_trl_IDs, trl_IDs_in_dir), 1, 'first') - 1;
%     start_trl_num = last_completed_trl_num + 1;
% end
% 
% % if this trial list has already been partially trained on, load last J
% 
% if start_trl_num > 1 % need to load last J if you had already started training on this set of trials!
%     prevMdl = load([rnnSubDir, dir([rnnSubDir, RNNname, ...
%         '_train_trl', num2str(train_trl_IDs(last_completed_trl_num)), '_num', num2str(last_completed_trl_num) '.mat']).name]);
%     prevJ = prevMdl.RNN.mdl.J;
%     prev_trl_ID = prevMdl.RNN.mdl.train_trl;
%     assert(prev_trl_ID == train_trl_IDs(last_completed_trl_num))
%     
%     % if done training, move on to testing
% elseif isempty(start_trl_num)
%     trainRNN = false;
%     prevMdl = load([rnnSubDir, dir([rnnSubDir, RNNname, ...
%         '_train_trl', num2str(train_trl_IDs(end)), '_num', num2str(length(train_trl_IDs)) '.mat']).name]);
%     prevJ = prevMdl.RNN.mdl.J;
%     prev_trl_ID = prevMdl.RNN.mdl.train_trl;
%     assert(prev_trl_ID == train_trl_IDs(end))
% end

[train_trl_IDs, test_trl_IDs, nTrlsTrain, nTrlsTest, start_trl_num, prevJ, trainRNN] ...
    = get_train_test_lists_and_progress(rnnDir, rnnSubDir, RNNname, nTrls, nTrlsIncluded);
%% TRAINING

if trainRNN == true
    
    % initialize outputs
    stdData = zeros(1, nTrls);
    JTrls = NaN(nUnits, nUnits, nTrls);
    
    trl_num = start_trl_num;
    
    for iTrlID = train_trl_IDs(start_trl_num : end) % startTrl : nTrls % TO DO 9/1/2022 CHANGE FOR LOOP TO DO TRAIN TRIALS DIFFERNETLY
        
        clear fittedConsJ
        fprintf('\n')
        disp([RNNname, ': training trial ', num2str(iTrlID), ', #=', num2str(trl_num), '.'])
        tic
        
        iStart = fixOnInds(iTrlID); % start of trial
        iStop = fixOnInds(iTrlID + 1) - 1; % right before start of next trial
        
        inputs = exp_data(:, iStart : iStop - 1);
        targets = exp_data(:, iStart + 1 : iStop);
        
        tData = allPossTS(iStart : iStop); % timeVec for current data
        tRNN = tData(1) : dtRNN : tData(end); % timevec for RNN
        
        % get fixed sample time points for subsampling J's and Rs
        sampleTimePoints = round(linspace(stimTimeInds(iTrlID), stimTimeInds(iTrlID) + minLen, nSamples));
        
        % set up white noise inputs (from CURBD)
        iWN = ampWN * randn( nUnits, length(tRNN) );
        inputWN = ones(nUnits, length(tRNN));
        
        for tt = 2 : length(tRNN)
            inputWN(:, tt) = iWN(:, tt) + (inputWN(:, tt - 1) - iWN(:, tt)) * exp( -(dtRNN / tauWN) );
        end
        
        inputWN = ampInWN * inputWN;
        
        % initialize DI matrix J
        if trainFromPrev && iTrlID == train_trl_IDs(start_trl_num) && exist('prevJ', 'var') && ~isempty(prevJ)
            J = prevJ;
        else
            if iTrlID == train_trl_IDs(1)
                J = g * randn(nUnits, nUnits) / sqrt(nUnits);
            else
                J = squeeze(JTrls(:, :, trl_num - 1));
            end
        end
        
        J0 = J;
        
        % get standard deviation of entire data that we are looking at
        stdData(trl_num)  = std(reshape(inputs(iTarget, :), length(iTarget) * (length(tData) - 1), 1));
        
        % get indices for each sample of model data for getting pVar
        iModelSample = zeros(length(tData) - 1, 1);
        
        for i = 1 : length(tData) - 1
            [~, iModelSample(i)] = min(abs(tData(i) - tRNN));
        end
        
        % initialize some others
        R = zeros(nUnits, length(tRNN)); % rate matrix - firing rates of neurons
        chi2 = zeros(1, nRunTrain);
        pVars = zeros(1, nRunTrain);
        JR = zeros(nUnits, length(tRNN)); % z(t) for the output readout unit
        
        % initialize learning update matrix (see Sussillo and Abbot, 2009)
        PJ = alpha * eye(nUnits); % dim are pN x pN where p=fraction of neurons to modify - here it's all of them
        
        if plotStatus
            f = figure('Position', [100 100 1800 600]);
        end
        
        %% training
        
        JLearn = NaN(nUnits, nUnits, size(inputs, 2));
        
        % loop through training runs
        for nRun = 1 : nRunTrain
            
            H = inputs(:, 1); % set initial condition to match target data
            R(:, 1) = nonlinearity(H); % convert to currents through nonlinearity
            
            tLearn = 0; % keeps track of current time
            iLearn = 1; % keeps track of last data point learned
            
            for tt = 2 : length(tRNN) % why start from 2?
                
                tLearn = tLearn + dtRNN; % index in actual RNN time
                
                R(:, tt) = nonlinearity(H); % compute next RNN step
                JR(:, tt) = J * R(:, tt) + inputWN(:, tt);
                H = H + dtRNN * (-H + JR(:, tt)) / tauRNN; % p much equivalent to: H + (dtRNN / tauRNN) * (-H + JR(:, tt));
                
                % update J if the RNN time coincides with a data point
                if tLearn >= dtData
                    tLearn = 0;
                    
                    error = R(1 : nUnits, tt) - targets(1 : nUnits, iLearn); % note: targets(:, iLearn) == inputs(:, iLearn + 1)
                    chi2(nRun) = chi2(nRun) + mean(error.^2); % update chi2 using this error
                    
                    iLearn = iLearn + 1; % update learning index
                    
                    if (nRun <= nRunTrain)
                        
                        k = PJ * R(iTarget, tt); % N x 1 (update term: sq mdl activity)
                        rPr = R(iTarget, tt)' * k; % scalar; inv xcorr of ntwk firing rates use xcorr bc when you square something it magnifies the sensitivity to changes
                        c = 1 / (1 + rPr); % tune learning rate by looking at magnitude of model activity. if big changes in FRs and wanna move quickly/take big steps (aka momentum). as get closer, don't wanna overshoot, so take smaller steps
                        PJ = PJ - c * (k * k'); % use squared firing rates (R * R^T) to update PJ - maybe momentum effect?
                        J(1 : nUnits, iTarget) = J(1 : nUnits, iTarget) - c * error(1 : nUnits, :) * k'; % update J (pre-syn wts adjusted according to post-syn target)
                        
                        if nRun == nRunTrain
                            JLearn(:, :, iLearn) = J; % thenwhen nRun == nRunTrain, compare JLearn2 and JLearn
                        end
                    end
                    
                end
            end
            
            rModelSample = R(iTarget, iModelSample);
            pVar = 1 - ( norm(targets(iTarget,:) - rModelSample, 'fro' ) / ( sqrt(length(iTarget) * (length(tData)-1)) * stdData(trl_num)) ).^2;
            pVars(nRun) = pVar;
            
            % if final run, save JLearn
            if nRun == nRunTrain
                fittedConsJ = JLearn(:, :, sampleTimePoints);
            end
            
            % plot
            if plotStatus
                clf(f);
                idx = randi(nUnits);
                subplot(2,4,1); hold on;
                imagesc(targets(iTarget, :)); colormap(jet), colorbar;
                axis tight; set(gca, 'clim', [0 1], 'Box', 'off', 'TickDir', 'out', 'FontSize', 14),
                title('targets')
                subplot(2,4,2); hold on;
                imagesc(R(iTarget, :)); colormap(jet), colorbar;
                axis tight; set(gca, 'clim', [0 1], 'Box', 'off', 'TickDir', 'out', 'FontSize', 14)
                title('model');
                subplot(2, 4, [3 4 7 8]); hold all;
                plot(tRNN, R(iTarget(idx), :), 'linewidth', 1.5);
                plot(tData(2 : end), targets(iTarget(idx), :), 'linewidth', 1.5);
                axis tight; set(gca, 'ylim', [-0.1 1], 'Box','off', 'TickDir', 'out', 'FontSize', 14)
                ylabel('activity'); xlabel('time (s)'),
                legend('model', 'target', 'location', 'eastoutside')
                title(['run ', num2str(nRun)])
                subplot(2, 4, 5); hold on;
                plot(pVars(1 : nRun)); ylabel('pVar');
                set(gca, 'ylim', [-0.1 1], 'Box', 'off', 'TickDir', 'out', 'FontSize', 14);
                title(['current pVar=', num2str(pVars(nRun), '%.3f')])
                subplot(2, 4, 6); hold on;
                plot(chi2(1 : nRun)); ylabel('chi2');
                set(gca, 'ylim', [-0.1 1], 'Box', 'off','TickDir', 'out', 'FontSize', 14);
                title(['current chi2=', num2str(chi2(nRun), '%.3f')])
                drawnow;
            end
        end
        
        % get elapsed time
        toc
        
        % save J for next trial (in trl_num (indices) order
        JTrls(:, :, trl_num) = J;
        
        % package up and save outputs at the end of training for each link
        RNN = struct;

        currentParams = struct( ...
            'doSmooth',             doSmooth, ...
            'smoothWidth',          smoothWidth, ...
            'meanSubtract',         meanSubtract, ...
            'doSoftNorm',           doSoftNorm, ...
            'normByRegion',         normByRegion, ...
            'rmvOutliers',          rmvOutliers, ...
            'outliers',             outliers, ...
            'exp_data_path',        [rnnDir 'exp_data' filesep, RNNname, '_exp_data.mat'], ...
            'dtFactor',             dtFactor, ...
            'g',                    g, ...
            'alpha',                alpha, ...
            'tauRNN',               tauRNN, ...
            'tauWN',                tauWN, ...
            'ampInWN',              ampInWN, ...
            'nRunTrain',            nRunTrain, ...
            'nonlinearity',         nonlinearity, ...
            'nUnits',               nUnits, ...
            'nTrls',                nTrls, ...
            'nSets',                nSets, ...
            'arrayUnit',            {arrayUnit}, ...
            'arrayRegions',         {arrayRgns});
        
        if trl_num == 1 % untested
            RNN.params = currentParams;
        else
            RNN.params = [];
        end
        
        RNN.mdl = struct( ...
            'train_trl',            iTrlID, ...
            'trl_num',              trl_num, ...
            'setID',                setID(iTrlID), ...
            'RMdlSample',           rModelSample, ...
            'tRNN',                 [], ...
            'dtRNN',                dtRNN, ...
            'exp_data',             [], ...
            'tData',                [], ...
            'dtData',               dtData, ...
            'J',                    J, ...
            'J0',                   J0, ...
            'fittedConsJ',          fittedConsJ, ...
            'sampleTimePoints',     sampleTimePoints, ...
            'chi2',                 chi2, ...
            'pVars',                pVars, ...
            'stdData',              stdData(trl_num), ...
            'inputWN',              [], ...
            'iTarget',              iTarget, ...
            'iNonTarget',           iNonTarget);
        
        if saveMdl
            save([rnnSubDir, RNNname, '_train_trl', num2str(iTrlID), '_num', num2str(trl_num) '.mat'], 'RNN', '-v7.3')
        end
        
        clear RNN fittedConsJ
        trl_num = trl_num + 1;
        close all;
    end
else
    %% TESTING
    
    % for train/test comparison
    all_pVars = cell2mat(arrayfun(@(iTrl) getfield(load([rnnSubDir, dir([rnnSubDir, RNNname, ...
        '_train_trl', num2str(train_trl_IDs(iTrl)), ...
        '_num', num2str(iTrl) '.mat']).name]), 'RNN', 'mdl', 'pVars'), 1 : nTrlsTrain, 'un', 0)');
    all_final_pVars = all_pVars(:, end);
    mean_pVar_train = mean(all_final_pVars);
    
    all_chi2 = cell2mat(arrayfun(@(iTrl) getfield(load([rnnSubDir, dir([rnnSubDir, RNNname, ...
        '_train_trl', num2str(train_trl_IDs(iTrl)), ...
        '_num', num2str(iTrl) '.mat']).name]), 'RNN', 'mdl', 'chi2'), 1 : nTrlsTrain, 'un', 0)');
    all_final_chi2 = all_chi2(:, end);
    mean_chi2_train = mean(all_final_chi2);
    
    testCostaRNNPredict;
end

end






