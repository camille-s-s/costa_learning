function RNN = fit_costa_RNN_prediction(RNNname, ...
    allSpikes, allPossTS, allEvents, arrayUnit, T, arrayRgns, params)
% Largely identical to fitCostaRNN except pre-processing steps.
%
% OLD VERSION 1. smooth 2. soft normalize 3. scale to [0 1] 4. remove outliers
%
% THIS VERSION 1. smooth 2. remove outliers 3. mean subtract for each neuron (maybe) 4. soft normalize AND ALSO
% (8/31/22) TESTING PREDICTION A LA KARPATHY

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
% tauWN           = 0.1;                  % decay constant on filtered white noise inputs
% ampInWN         = 0.001;                % input amplitude of filtered white noise

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
% nSamples        = 21;                   % #samples to save from consecutive fitted Js
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

%% preprocess targets by smoothing, normalizing, re-scaling, and outlier removing, or load previous

if ~isfile([rnnDir 'exp_data' filesep, RNNname, '_exp_data.mat'])
    
    [exp_data, outliers, arrayUnit, arrayRgns, fixOnInds, stimTimeInds, nTrls, nTrlsPerSet, nSets, setID] ...
        = preprocess_data_for_RNN_training(allSpikes, allEvents, T, ...
        doSmooth, rmvOutliers, meanSubtract, doSoftNorm, ...
        smoothWidth, dtData, arrayUnit, arrayRgns);
    
    save([rnnDir 'exp_data' filesep, RNNname, '_exp_data.mat'],  'exp_data', 'outliers', 'fixOnInds', 'stimTimeInds', 'nTrls', 'nTrlsPerSet', 'nSets', 'setID', 'arrayUnit', 'arrayRgns', '-v7.3')
else
    load([rnnDir 'exp_data' filesep, RNNname, '_exp_data.mat'], 'exp_data', 'outliers', 'fixOnInds', 'stimTimeInds', 'nTrls', 'nTrlsPerSet', 'nSets', 'setID', 'arrayUnit', 'arrayRgns');
end

clearvars allEvents prevMdls allSpikes T

% define minimimum trial length so all sampled Js are in same timescale
if isnan(minLen)
    minLen = min(diff(fixOnInds)); % shortest trial (from fixation to next fixation)
end

nUnits = size(exp_data, 1);

%% generate train trial ID list (if trained via multiple fcn calls)

% define train/test split
nTrlsIncluded = 100;
[train_trl_IDs, test_trl_IDs, nTrlsTrain, nTrlsTest, start_trl_num, prevJ, trainRNN] ...
    = get_train_test_lists_and_progress(rnnDir, rnnSubDir, RNNname, nTrls, nTrlsIncluded);

%%
if trainRNN % TRAIN
    
    % initialize outputs
    stdData = zeros(1, nTrls);
    
    trl_num = start_trl_num;
    
    for iTrlID = train_trl_IDs(start_trl_num : end)
        
        fprintf('\n')
        fprintf([RNNname, ': training trial ', num2str(iTrlID), ', #=', num2str(trl_num), '...'])
        tic
        
        iStart = fixOnInds(iTrlID); % start of trial
        iStop = fixOnInds(iTrlID + 1) - 1; % right before start of next trial
        inputs = exp_data(:, iStart : iStop - 1);
        targets = exp_data(:, iStart + 1 : iStop);
        tData = allPossTS(iStart : iStop); nsp_Data = length(tData); % timeVec for current data
        tRNN = tData(1) : dtRNN : tData(end); nsp_RNN = length(tRNN); % timevec for RNN
        
        % initialize DI matrix J
        if trainFromPrev && iTrlID == train_trl_IDs(start_trl_num) && exist('prevJ', 'var') && ~isempty(prevJ)
            fprintf('Using prevJ from last trained trial. ')
            J = prevJ;
        else
            if iTrlID == train_trl_IDs(1)
                J = g * randn(nUnits, nUnits) / sqrt(nUnits);
            end
        end
        
        J0 = J;
        
        % get standard deviation of entire data that we are looking at
        stdData(trl_num)  = std(reshape(inputs, nUnits * (nsp_Data - 1), 1));
        
        % get indices for each sample of model data for getting pVar
        iModelSample = zeros(nsp_Data - 1, 1);
        
        for i = 1 : nsp_Data - 1
            [~, iModelSample(i)] = min(abs(tData(i) - tRNN));
        end
        
        % initialize midputs
        R = zeros(nUnits, nsp_RNN); % rate matrix - firing rates of neurons
        chi2 = zeros(1, nRunTrain);
        pVars = zeros(1, nRunTrain);
        JR = zeros(nUnits, nsp_RNN); % z(t) for the output readout unit
        
        % initialize learning update matrix (see Sussillo and Abbot, 2009)
        PJ = alpha * eye(nUnits); % dim are pN x pN where p=fraction of neurons to modify - here it's all of them
        
        if plotStatus
            f = figure('color', 'w', 'Position', [100 100 1800 600]);
        end
        
        %% loop through training runs
        
        for nRun = 1 : nRunTrain
            
            H = inputs(:, 1); % set initial condition to match target data
            % R(:, 1) = nonlinearity(H); % convert to currents through nonlinearity
            R(:, 1) = tanh(H); % convert to currents through nonlinearity
            
            tLearn = 0; % keeps track of current time
            iLearn = 1; % keeps track of last data point learned
            
            for t = 2 : nsp_RNN % why start from 2?
                
                tLearn = tLearn + dtRNN; % index in actual RNN time
                % R(:, t) = nonlinearity(H); % compute next RNN step
                R(:, t) = tanh(H); % compute next RNN step
                JR(:, t) = J * R(:, t) + inputs(:, iLearn); % 2022-09-27 CURRENTLY TESTING INCORPORATING INPUT!
                H = H + dtRNN * (-H + JR(:, t)) / tauRNN; % p much equivalent to: H + (dtRNN / tauRNN) * (-H + JR(:, tt));
                
                % update J if the RNN time coincides with a data point
                if tLearn >= dtData
                    
                    tLearn = 0;
                    % error = JR(1 : nUnits, t) - targets(1 : nUnits, iLearn); % 2022-09-27 CURRENTLY TESTING!
                    error = JR(:, t) - targets(:, iLearn); % IDENTICAL TO ABOVE BUT FASTER
                    chi2(nRun) = chi2(nRun) + mean(error.^2); % update chi2 using this error
                    iLearn = iLearn + 1; % update learning index
                    
                    if (nRun <= nRunTrain)
                        
                        k = PJ * R(:, t); % N x 1 (update term: sq mdl activity)
                        rPr = R(:, t)' * k; % scalar; inv xcorr of ntwk firing rates use xcorr bc when you square something it magnifies the sensitivity to changes
                        c = 1 / (1 + rPr); % tune learning rate by looking at magnitude of model activity. if big changes in FRs and wanna move quickly/take big steps (aka momentum). as get closer, don't wanna overshoot, so take smaller steps
                        PJ = PJ - c * (k * k'); % use squared firing rates (R * R^T) to update PJ - maybe momentum effect?
                        % J(1 : nUnits, :) = J(1 : nUnits, :) - c * error(1 : nUnits, :) * k'; % update J (pre-syn wts adjusted according to post-syn target)
                        J = J - c * error * k'; % identical to above but faster in current iteration
                    end
                end
            end
            
            rModelSample = R(:, iModelSample);
            pVar = 1 - ( norm(targets - rModelSample, 'fro' ) / ( sqrt(nUnits * (nsp_Data - 1)) * stdData(trl_num)) ).^2;
            pVars(nRun) = pVar;
            
            if plotStatus
                plot_costa_RNN_progress(f, nUnits, targets, R, tRNN, tData, nRun, pVars, chi2, trainRNN)
            end
        end
        
        % package up and save outputs
        RNN = make_RNN_struct(trainRNN, doSmooth, smoothWidth, meanSubtract, doSoftNorm, normByRegion, rmvOutliers, ...
            outliers, rnnDir, RNNname, dtFactor, g, alpha, tauRNN, [], [], [], nRunTrain, nonlinearity, ...
            nUnits, nTrls, nSets, arrayUnit, arrayRgns, iTrlID, trl_num, setID, rModelSample, [], [], [], ...
            dtRNN, dtData, J, J0, [], [], chi2, pVars, stdData);
        
        if saveMdl
            save([rnnSubDir, RNNname, '_train_trl', num2str(iTrlID), '_num', num2str(trl_num) '.mat'], 'RNN', '-v7.3')
        end
        
        clear RNN
        trl_num = trl_num + 1;
        close all;
        toc
    end
    
else % TEST
    
    % get convergence metrics so you can compare train/test
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
    
    stdData_test = zeros(1, length(test_trl_IDs));
    
    start_trl_num = 1;
    trl_num = start_trl_num;
    
    % if condition met showing we're at end of training, J is J from end of training full training set
    if ~exist('J', 'var') % at present you run testing and training separately (not in same script call)
        fprintf('Using J from last trained trial.')
        J_test = prevJ;
    else
        keyboard % just to catch potential J inconsistencies
    end
    
    pVars_test = zeros(nTrlsTest, nRunTest);
    final_chi2_test = zeros(nTrlsTest, nRunTest);
    
    for iTrlID = test_trl_IDs(start_trl_num : end)
        
        fprintf('\n')
        fprintf([RNNname, ': testing trial ', num2str(iTrlID), ', #=', num2str(trl_num), '...'])
        tic
        
        iStart = fixOnInds(iTrlID); % start of trial
        iStop = fixOnInds(iTrlID + 1) - 1; % right before start of next trial
        inputs = exp_data(:, iStart : iStop - 1);
        targets = exp_data(:, iStart + 1 : iStop);
        tData = allPossTS(iStart : iStop); nsp_Data = length(tData); % timeVec for current data
        tRNN = tData(1) : dtRNN : tData(end); nsp_RNN = length(tRNN); % timevec for RNN
        
        % get standard deviation of entire data that we are looking at
        stdData_test(trl_num)  = std(reshape(inputs, nUnits * (nsp_Data - 1), 1));
        
        % get indices for each sample of model data for getting pVar
        iModelSample = zeros(nsp_Data - 1, 1);
        
        for i = 1 : nsp_Data - 1
            [~, iModelSample(i)] = min(abs(tData(i) - tRNN));
        end
        
        % initialize midputs
        R_test = zeros(nUnits, nsp_RNN); % rate matrix - firing rates of neurons
        chi2_test = zeros(1, nsp_Data);
        JR_test = zeros(nUnits, nsp_RNN); % z(t) for the output readout unit
        
        if plotStatus
            f = figure('color', 'w', 'Position', [100 100 1900 700]);
        end
        
        %% loop through testing runs
        
        for nRun = 1 : nRunTest
            
            H = inputs(:, 1); % seed with start of test data set
            R_test(:, 1) = tanh(H); % convert to currents through nonlinearity
            
            tLearn = 0; % keeps track of current time
            iLearn = 1; % keeps track of last data point learned
            
            for t = 2 : nsp_RNN
                
                tLearn = tLearn + dtRNN; % index in actual RNN time
                R_test(:, t) = tanh(H); % compute next RNN step
                JR_test(:, t) = J_test * R_test(:, t) + inputs(:, iLearn); % 2022-09-27 CURRENTLY TESTING INCORPORATING INPUT!
                H = H + dtRNN * (-H + JR_test(:, t)) / tauRNN; % p much equivalent to: H + (dtRNN / tauRNN) * (-H + JR(:, tt));
                
                if tLearn >= dtData
                    tLearn = 0;
                    
                    error = JR_test(:, t) - targets(:, iLearn); % 2022-09-27 CURRENTLY TESTING!
                    chi2_test(iLearn) = mean(error .^ 2);
                    
%                     if iLearn == 1
%                         chi2_test(1, iLearn) = mean(error .^ 2);
%                     elseif iLearn > 1
%                         chi2_test(1, iLearn) = chi2_test(1, iLearn - 1) + mean(error .^ 2); % update chi2 using this error
%                     end
                    
                    iLearn = iLearn + 1; % update learning index
                end
            end
            
            rModelSample_test = R_test(:, iModelSample);
            pVar = 1 - ( norm(targets - rModelSample_test, 'fro' ) / ( sqrt(nUnits * (nsp_Data-1)) * stdData_test(trl_num)) ).^2;
            pVars_test(trl_num, nRun) = pVar;
            
            cumulative_chi2_test = cumsum(chi2_test);
                    
            if plotStatus
                plot_costa_RNN_progress(f, nUnits, targets, R_test, tRNN, tData, nRun, pVars_test, cumulative_chi2_test, trainRNN)
            end
        end
        
        % last cumulative chi2 of a given trial      
        final_chi2_test(trl_num) = cumulative_chi2_test(end); % chi2_test(nsp_Data - 1);
        
        % package up and save outputs
        RNN = make_RNN_struct(trainRNN, doSmooth, smoothWidth, meanSubtract, doSoftNorm, normByRegion, rmvOutliers, ...
            outliers, rnnDir, RNNname, dtFactor, g, alpha, tauRNN, [], [], [], nRunTrain, nonlinearity, ...
            nUnits, nTrls, nSets, arrayUnit, arrayRgns, iTrlID, trl_num, setID, rModelSample_test, [], [], [], ...
            dtRNN, dtData, J_test, [], [], [], cumulative_chi2_test, pVars_test, stdData_test);
        
        if saveMdl
            save([rnnSubDir, RNNname, '_test_trl', num2str(iTrlID), '_num', num2str(trl_num) '.mat'], 'RNN', '-v7.3')
        end
        
        clear RNN
        trl_num = trl_num + 1;
        close all;
        toc
    end
    
    % get convergence metrics so you can compare train/test
    mean_final_pVar_test = mean(pVars_test);
    mean_final_chi2_test = mean(final_chi2_test);
end

end





