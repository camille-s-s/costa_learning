function RNN = fit_costa_RNN_prediction(RNNname, ...
    allSpikes, allPossTS, allEvents, arrayUnit, T, arrayRgns, ntwk_flavor, params)
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
tauWN           = 0.1;                  % decay constant on filtered white noise inputs
ampInWN         = 0.001;                % input amplitude of filtered white noise

%% training params
alpha           = 1;                    % overall learning rate for regularizer
nonlinearity    = @tanh;                % inline function for nonlinearity
nRunTrain       = 1000;
nRunTest        = 1;                    % TO DO: IDK HOW MUCH TO TEST?!?!
trainFromPrev   = false;                % assume you're starting from beginning, but if not, feed in previous J
trainAllUnits   = true;                 % modify all units in network by default
trainRNN        = true;                 % true by default until it's not!
nTrlsIncluded   = 400;
nUnitsIncluded  = 25;
n_pred_steps    = 2; % each timestep = dtData (in this case currently 10ms)

%% output options
plotStatus      = true;
saveMdl         = true;

%% RNN

% overwrite defaults based on inputs
if exist('params', 'var')
    assignParams(who, params);
end

% define output directories
rnnDir          = ['~/Dropbox (BrAINY Crew)/costa_learning/models/', mouseVer, filesep];
rnnSubDir       = [rnnDir, RNNname(strfind(RNNname, '_') + 1 : end), filesep];

if ~isfolder(rnnSubDir)
    mkdir(rnnSubDir)
end
if ~isfolder([rnnDir, 'train_test_lists'])
    mkdir([rnnDir, 'train_test_lists'])
end
if ~isfolder([rnnDir 'exp_data' filesep]) % separate subdir for exp_data ( inputs=exp_data(t), targets=exp_data(t+1) )
    mkdir([rnnDir 'exp_data' filesep])
end

% set up final params
dtRNN           = dtData / dtFactor;    % time step (in s) for integration
ampWN           = sqrt( tauWN / dtRNN );

%% preprocess targets by smoothing, normalizing, re-scaling, and outlier removing, or load previous

if ~isfile([rnnDir 'exp_data' filesep, RNNname, '_exp_data.mat'])
    [exp_data, outliers, arrayUnit, arrayRgns, fixOnInds, stimTimeInds, nTrls, nTrlsPerSet, nSets, setID] ...
        = preprocess_data_for_RNN_training(allSpikes, allEvents, T, doSmooth, rmvOutliers, meanSubtract, doSoftNorm, ...
        smoothWidth, dtData, arrayUnit, arrayRgns);
    save([rnnDir 'exp_data' filesep, RNNname, '_exp_data.mat'],  'exp_data', 'outliers', 'fixOnInds', 'stimTimeInds', 'nTrls', 'nTrlsPerSet', 'nSets', 'setID', 'arrayUnit', 'arrayRgns', '-v7.3')
else
    load([rnnDir 'exp_data' filesep, RNNname, '_exp_data.mat'], 'exp_data', 'outliers', 'fixOnInds', 'stimTimeInds', 'nTrls', 'nTrlsPerSet', 'nSets', 'setID', 'arrayUnit', 'arrayRgns');
end

clearvars allEvents prevMdls allSpikes T
nUnits = size(exp_data, 1);

%% generate train trial ID split and list (if trained via multiple fcn calls)

[train_trl_IDs, test_trl_IDs, nTrlsTrain, nTrlsTest, start_trl_num, prevJ, trainRNN, iPredict] = get_train_test_lists_and_progress(rnnDir, rnnSubDir, RNNname, nTrlsIncluded, nUnits, trainAllUnits, nUnitsIncluded);
nPredict = length(iPredict);
iNonTarget = find(~ismember(1 : nUnits, iPredict)); % sanity: assert(isempty(intersect(iNonTarget, iPredict)))

%% snipping out most units

% previous way: put 0s in nonplastic rows
exp_data_tmp = zeros(size(exp_data));
exp_data_tmp(iPredict, :) = exp_data(iPredict, :);

% for non included units, cut out 
nRmv = nUnits - (1 * nPredict);
rmvdNonTargetUnits = randperm(length(iNonTarget), nRmv);
exp_data_tmp(iNonTarget(rmvdNonTargetUnits), :) = []; % slice out zero rows at iNonTarget indices
exp_data = exp_data_tmp; clearvars exp_data_tmp

exp_data = exp_data(iPredict, :);

% old indexing now useless (unless you save rmvdNonTargetUnits, which I don't yet) - should fix
nUnits = size(exp_data, 1); % sloppy overwrite
iPredict = find(any(exp_data, 2));

%% cut down variable size for memory saving

training_inputs = cell(nTrlsTrain, 1);
training_targets = cell(nTrlsTrain, 1);
training_tData = cell(nTrlsTrain, 1);
trl_num = 1;
for iTrlID = train_trl_IDs(1 : end)
    iStart = fixOnInds(iTrlID); % start of trial
    iStop = fixOnInds(iTrlID + 1) - 1; % right before start of next trial
    training_inputs{trl_num} = exp_data(:, iStart : iStop - n_pred_steps);
    training_targets{trl_num} = exp_data(:, iStart + n_pred_steps : iStop);
    training_tData{trl_num} = allPossTS(iStart : iStop);
    trl_num = trl_num + 1;
end
testing_inputs = cell(nTrlsTest, 1);
testing_targets = cell(nTrlsTest, 1);
testing_tData = cell(nTrlsTest, 1);
trl_num = 1;
for iTrlID = test_trl_IDs(1 : end)
    iStart = fixOnInds(iTrlID); % start of trial
    iStop = fixOnInds(iTrlID + 1) - 1; % right before start of next trial
    testing_inputs{trl_num} = exp_data(:, iStart : iStop - n_pred_steps);
    testing_targets{trl_num} = exp_data(:, iStart + n_pred_steps : iStop);
    testing_tData{trl_num} = allPossTS(iStart : iStop);
    trl_num = trl_num + 1;
end
tmp_vals = [cell2mat(training_targets'), cell2mat(testing_targets')];
[~, lower_FR_thresh, upper_FR_thresh] = isoutlier(tmp_vals(:), 'percentiles', [0.1 99.9]);
clearvars exp_data allPossTS

%%
if trainRNN % TRAIN
    
    % initialize outputs
    stdData = zeros(1, nTrls);
    trl_num = start_trl_num;
    
    for iTrlID = train_trl_IDs(start_trl_num : end)
        fprintf('\n')
        fprintf([RNNname, ': training trial ', num2str(iTrlID), ', #=', num2str(trl_num), '...\n'])
        tic
        inputs = training_inputs{trl_num}; % exp_data(:, iStart : iStop - 1);
        targets = training_targets{trl_num}; % exp_data(:, iStart + 1 : iStop);
        tData = training_tData{trl_num}; % allPossTS(iStart : iStop);
        nsp_Data = length(tData); % timeVec for current data
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
        J0 = J; % save initial J
        stdData(trl_num)  = std(reshape(inputs(iPredict, :), nPredict * (nsp_Data - n_pred_steps), 1)); % std of entire data that we are looking at
        
        iMdlSample = zeros(nsp_Data - n_pred_steps, 1); % get indices for each sample of model data
        for i = 1 : nsp_Data - n_pred_steps
            [~, iMdlSample(i)] = min(abs(tData(i) - tRNN));
        end
        
        white_noise = get_frozen_input_WN(nUnits, ampWN, tauWN, ampInWN, nsp_RNN, dtRNN); % set up white noise inputs (from CURBD)
        
        % initialize midputs
        R = zeros(nUnits, nsp_RNN); % rate matrix - firing rates of neurons
        sum_MSE_over_runs = NaN(1, nRunTrain);
        mean_MSE_over_runs = NaN(1, nRunTrain);
        pVars = zeros(1, nRunTrain);
        JR = zeros(nUnits, nsp_RNN); % z(t) for the output readout unit
        
        % initialize learning update matrix (see Sussillo and Abbot, 2009)
        PJ = alpha * eye(nUnits); % dim are pN x pN where p=fraction of neurons to modify - here it's all of them
        % PJ = alpha * eye(nPredict); % dim are pN x pN where p=fraction of neurons to modify - here it's all of them
        
        if plotStatus && ismember(trl_num, floor(linspace(1, nTrlsTrain, 6)))
            f = figure('color', 'w', 'Position', [100 100 1900 750]);
            axNum = 0;
            f2 = figure('color', 'w', 'Position', [100 100 1900 500]);
            axs = arrayfun( @(i) subplot(1, 4, i, 'NextPlot', 'add', 'box', 'off', 'tickdir', 'out', 'fontsize', 12, 'fontweight', 'bold'), 1 : 4);
            paramText = sprintf(['TRAINING\nkernel = \t', num2str(smoothWidth), '\nn_t_r_l_s = \t', num2str(length([train_trl_IDs, test_trl_IDs])), '\ng = \t', num2str(g), '\nP_0 = \t', num2str(alpha), '\ntau_R_N_N = \t', num2str(tauRNN), '\ntau_W_N = \t', num2str(tauWN), '\nw_W_N = \t', num2str(ampInWN), '\nn_i_t_e_r = \t', num2str(nRunTrain), '\ndt_D_a_t_a = \t', num2str(dtData), '\ndt_R_N_N = \t', num2str(dtRNN)  ]); % this at the end!
            text(axs(4), 0.25 * range(get(axs(4), 'xlim')), 0.75 * range(get(axs(4), 'ylim')), paramText, 'fontsize', 14, 'fontweight', 'bold', 'horizontalalignment', 'center')
            set(axs(4), 'xtick', '', 'ytick', '')
            % set up for histogram
            nBins = 25;
            max_count = nsp_Data - n_pred_steps;
        end
        
        %% loop through training runs
        for nRun = 1 : nRunTrain
            H = inputs(:, 1); % set initial condition to match target data % OR SHOULD IT BE INPUTS(iPredict, 1)?!
            R(:, 1) = tanh(H); % convert to currents through nonlinearity
            tLearn = 0; % keeps track of current time
            iLearn = 1; % keeps track of last data point learned
            MSE_over_steps = zeros(1, nsp_Data - n_pred_steps);
            for t = 2 : nsp_RNN - (n_pred_steps - 1) * dtFactor % 2 : nsp_RNN usually. this range will be identical for n_pred_steps < 2 as before
                tLearn = tLearn + dtRNN; % index in actual RNN time
                R(:, t) = tanh(H); % generate output / compute next RNN step
                switch ntwk_flavor
                    case 'descriptive'
                        variable_term = white_noise(:, t);
                    case '1step'
                        variable_term = inputs(:, iLearn);
                    case 'generative'
                        if iLearn < 2 % && t == iMdlSample(iLearn) + 1
                            variable_term = R(:, 1);
                        elseif iLearn >= 2 % && t == iMdlSample(iLearn) + 1
                            variable_term = R(:, iMdlSample(iLearn - 1) + 1); % inputs(:, iLearn - 1);% R(:, iMdlSample(iLearn - 1) + 1);
                        end
                end
                JR(:, t) = J * R(:, t) + variable_term;
                H = H + dtRNN * (-H + JR(:, t)) / tauRNN; % "hidden" activity updated w the update term, p much equivalent to: H + (dtRNN / tauRNN) * (-H + JR(:, tt));
                % update J if the RNN time coincides with a data point
                if tLearn >= dtData
                    tLearn = 0;
                    switch ntwk_flavor
                        case 'descriptive'
                            error = R(:, t) - inputs(:, iLearn); % description
                        case {'1step', 'generative'}
                            error = R(:, t) - targets(:, iLearn); % prediction
                    end
                    MSE_over_steps(iLearn) = mean(error .^ 2);
                    iLearn = iLearn + 1; % update learning index
                    if (nRun <= nRunTrain)
                        k = PJ * R(:, t); % N x 1 (update term: sq mdl activity)
                        rPr = R(:, t)' * k; % scalar; inv xcorr of ntwk firing rates use xcorr bc when you square something it magnifies the sensitivity to changes
                        c = 1 / (1 + rPr); % tune learning rate by looking at magnitude of model activity. if big changes in FRs and wanna move quickly/take big steps (aka momentum). as get closer, don't wanna overshoot, so take smaller steps
                        PJ = PJ - c * (k * k'); % use squared firing rates (R * R^T) to update PJ - maybe momentum effect?
                        J(:, :) = J(:, :) - c * error * k'; % update J (pre-syn wts adjusted according to post-syn target)
                    end
                end
            end
            rModelSample = R(iPredict, iMdlSample);
            pVar = 1 - ( norm(targets(iPredict, :) - rModelSample, 'fro' ) / ( sqrt(nPredict * (nsp_Data - n_pred_steps)) * stdData(trl_num)) ).^2;
            pVars(nRun) = pVar;
            mean_MSE_over_runs(nRun) = mean(MSE_over_steps);
            sum_MSE_over_runs(nRun) = sum(MSE_over_steps);
            if ismember(nRun, [1 250 nRunTrain]) && ismember(trl_num, floor(linspace(1, nTrlsTrain, 6)))
                axNum = axNum + 1;
                plot_costa_RNN_param_comparisons(f2, axs, [RNNname(5 : end), ' (ID: ', num2str(iTrlID), ' / # ', num2str(trl_num), '):'], nPredict, targets(iPredict, :), R(iPredict, iMdlSample), tRNN(:, iMdlSample), tData(n_pred_steps + 1 : end), nRun, pVars, MSE_over_steps, axNum)
            end
        end
        if plotStatus
            if ismember(trl_num, floor(linspace(1, nTrlsTrain, 6)))
                set(0, 'currentfigure', f2);
                print('-dtiff', '-r400', [rnnSubDir, RNNname, '_train_trl', num2str(iTrlID), '_num', num2str(trl_num), '_runs'])
                set(0, 'currentfigure', f);
                fTitle = ['[', RNNname, ']: train trial ID ', num2str(iTrlID), ' (#', num2str(trl_num), ')'];
                plot_costa_RNN_progress(f, nUnits, targets(:, :), R(:, iMdlSample), tRNN(:, iMdlSample), tData(n_pred_steps + 1 : end), nRun, pVars, sum_MSE_over_runs, trainRNN, '')
                fAxs = findall(f, 'type', 'axes'); set(fAxs(2), 'ylim', [-0.2 1]);
                print('-dtiff', '-r400', [rnnSubDir, RNNname, '_train_trl', num2str(iTrlID), '_num', num2str(trl_num), '_overall'])
                plot_comparative_FR_hists_each_unit(nBins, nUnits, nPredict, max_count, lower_FR_thresh, upper_FR_thresh, R(:, iMdlSample), targets, fTitle)
                print('-dtiff', '-r400', [rnnSubDir, RNNname, '_train_trl', num2str(iTrlID), '_num', num2str(trl_num), '_FR_comparison_hists'])
            end
        end
        
        % package up and save outputs
        [RNN] = make_RNN_struct(trainRNN, doSmooth, smoothWidth, meanSubtract, doSoftNorm, normByRegion, rmvOutliers, ...
            outliers, rnnDir, RNNname, dtFactor, g, alpha, tauRNN, tauWN, ampInWN, ampWN, nRunTrain, nonlinearity, ...
            nUnits, nTrls, nSets, arrayUnit, arrayRgns, iTrlID, trl_num, setID, R(:, iMdlSample), [], [], [], ...
            dtRNN, dtData, J, J0, [], [], sum_MSE_over_runs, mean_MSE_over_runs, pVars, stdData, iPredict);
        
        if saveMdl
            save([rnnSubDir, RNNname, '_train_trl', num2str(iTrlID), '_num', num2str(trl_num) '.mat'], 'RNN', '-v7.3')
        end
        
        clear RNN
        trl_num = trl_num + 1;
        close all;
        toc
    end
    
else % TEST
    
    all_iPredicts = cell2mat(arrayfun(@(iTrl) getfield(load([rnnSubDir, dir([rnnSubDir, RNNname, ...
        '_train_trl', num2str(train_trl_IDs(iTrl)), ...
        '_num', num2str(iTrl) '.mat']).name]), 'RNN', 'mdl', 'iPredict'), 1 : nTrlsTrain, 'un', 0)');
    assert(isequal(unique(all_iPredicts, 'rows'), iPredict)) % sanity check: all iPredicts for all trials in training set must be identical and equal to iPredict
    
    % get convergence metrics so you can compare train/test
    all_pVars = cell2mat(arrayfun(@(iTrl) getfield(load([rnnSubDir, dir([rnnSubDir, RNNname, ...
        '_train_trl', num2str(train_trl_IDs(iTrl)), ...
        '_num', num2str(iTrl) '.mat']).name]), 'RNN', 'mdl', 'pVars'), 1 : nTrlsTrain, 'un', 0)');
    all_mean_MSE_over_runs = cell2mat(arrayfun(@(iTrl) getfield(load([rnnSubDir, dir([rnnSubDir, RNNname, ...
        '_train_trl', num2str(train_trl_IDs(iTrl)), ...
        '_num', num2str(iTrl) '.mat']).name]), 'RNN', 'mdl', 'mean_MSE_over_runs'), 1 : nTrlsTrain, 'un', 0)');
    all_cum_MSE_over_runs = cell2mat(arrayfun(@(iTrl) getfield(load([rnnSubDir, dir([rnnSubDir, RNNname, ...
        '_train_trl', num2str(train_trl_IDs(iTrl)), ...
        '_num', num2str(iTrl) '.mat']).name]), 'RNN', 'mdl', 'cumulative_MSE_over_runs'), 1 : nTrlsTrain, 'un', 0)');
    all_final_pVars = all_pVars(:, end);
    all_final_mean_MSE = all_mean_MSE_over_runs(:, end);
    all_final_cum_MSE = all_cum_MSE_over_runs(:, end);
    
    avg_final_pVar_train = mean(all_final_pVars);
    avg_final_mean_MSE_train = mean(all_final_mean_MSE);
    avg_final_sum_MSE_train = mean(all_final_cum_MSE);
    
    stdData_test = zeros(1, nTrlsTest);
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
    sum_MSE_over_runs_test = zeros(nTrlsTest, nRunTest);
    mean_MSE_over_runs_test = zeros(nTrlsTest, nRunTest);
    
    for iTrlID = test_trl_IDs(start_trl_num : end)
        fprintf('\n')
        fprintf([RNNname, ': testing trial ', num2str(iTrlID), ', #=', num2str(trl_num), '...'])
        tic
        inputs = testing_inputs{trl_num}; % exp_data(:, iStart : iStop - 1);
        targets = testing_targets{trl_num}; % exp_data(:, iStart + 1 : iStop);
        tData = testing_tData{trl_num}; % allPossTS(iStart : iStop);
        nsp_Data = length(tData);
        tRNN = tData(1) : dtRNN : tData(end); nsp_RNN = length(tRNN);
        stdData_test(trl_num)  = std(reshape(inputs(iPredict, :), nPredict * (nsp_Data - n_pred_steps), 1));
        iMdlSample_test = zeros(nsp_Data - n_pred_steps, 1);
        
        for i = 1 : nsp_Data - n_pred_steps
            [~, iMdlSample_test(i)] = min(abs(tData(i) - tRNN));
        end
        
        white_noise = get_frozen_input_WN(nUnits, ampWN, tauWN, ampInWN, nsp_RNN, dtRNN); % set up white noise inputs (from CURBD)
        
        % initialize midputs
        R_test = zeros(nUnits, nsp_RNN);
        JR_test = zeros(nUnits, nsp_RNN);
        
        if plotStatus && ismember(trl_num, floor(linspace(1, nTrlsTest, 6))) % plot progress as we tweak parameters
            f = figure('color', 'w', 'Position', [100 100 1900 750]);
            axNum = 0;
            f2 = figure('color', 'w', 'Position', [100 100 1900 500]);
            axs = arrayfun( @(i) subplot(1, 4, i, 'NextPlot', 'add', 'box', 'off', 'tickdir', 'out', 'fontsize', 12, 'fontweight', 'bold'), 1 : 4);
            paramText = sprintf(['TESTING\nkernel = \t', num2str(smoothWidth), '\nn_t_r_l_s = \t', num2str(length([train_trl_IDs, test_trl_IDs])), '\ng = \t', num2str(g), '\nP_0 = \t', num2str(alpha), '\ntau_R_N_N = \t', num2str(tauRNN), '\ntau_W_N = \t', num2str(tauWN), '\nw_W_N = \t', num2str(ampInWN), '\nn_i_t_e_r = \t', num2str(nRunTrain), '\ndt_D_a_t_a = \t', num2str(dtData), '\ndt_R_N_N = \t', num2str(dtRNN)  ]);
            text(axs(4), 0.25 * range(get(axs(4), 'xlim')), 0.75 * range(get(axs(4), 'ylim')), paramText, 'fontsize', 14, 'fontweight', 'bold', 'horizontalalignment', 'center')
            set(axs(4), 'xtick', '', 'ytick', '')
        end
        
        %% loop through testing runs
        for nRun = 1 : nRunTest
            H = inputs(:, 1); % !!! can be initialized with some random gaussian? nUnits x 1 random sample
            R_test(:, 1) = tanh(H);
            tLearn = 0;
            iLearn = 1;
            MSE_over_steps_test = zeros(1, nsp_Data - n_pred_steps);
            for t = 2 : nsp_RNN - (n_pred_steps - 1) * dtFactor % 2 : nsp_RNN
                tLearn = tLearn + dtRNN;
                R_test(:, t) = tanh(H); % generate output
                switch ntwk_flavor
                    case 'descriptive'
                        variable_term = white_noise(:, t);
                    case '1step'
                        variable_term = inputs(:, iLearn);
                    case 'generative'
                        if iLearn < 2 % && t == iMdlSample(iLearn) + 1
                            variable_term = R_test(:, 1);
                        elseif iLearn >= 2 % && t == iMdlSample(iLearn) + 1
                            variable_term = R_test(:, iMdlSample_test(iLearn - 1) + 1); % inputs(:, iLearn - 1);% R(:, iMdlSample(iLearn - 1) + 1);
                        end
                end                
                JR_test(:, t) = J_test * R_test(:, t) + variable_term;
                H = H + dtRNN * (-H + JR_test(:, t)) / tauRNN;
                if tLearn >= dtData
                    tLearn = 0;
                    switch network_flavor
                        case 'descriptive'
                            error = R_test(:, t) - inputs(:, iLearn); % description
                        case {'1step', 'generative'}
                            error = R_test(:, t) - targets(:, iLearn); % prediction
                    end
                    MSE_over_steps_test(iLearn) = mean(error .^ 2);
                    iLearn = iLearn + 1;
                end
            end
            
            rModelSample_test = R_test(iPredict, iMdlSample_test);
            pVar = 1 - ( norm(targets(iPredict, :) - rModelSample_test, 'fro' ) / ( sqrt(nPredict * (nsp_Data - n_pred_steps)) * stdData_test(trl_num)) ) .^ 2;
            pVars_test(trl_num, nRun) = pVar;
            mean_MSE_over_runs_test(trl_num, nRun) = mean(MSE_over_steps_test);
            sum_MSE_over_runs_test(trl_num, nRun) = sum(MSE_over_steps_test);
            
            if ismember(nRun, [1 250 nRunTrain]) && ismember(trl_num, floor(linspace(1, nTrlsTest, 6)))
                axNum = axNum + 1;
                plot_costa_RNN_param_comparisons(f2, axs, [RNNname(5 : end), ' (ID: ', num2str(iTrlID), ' / # ', num2str(trl_num), '):'], nPredict, targets(iPredict, :), R_test(iPredict, iMdlSample_test), tRNN(:, iMdlSample_test), tData(n_pred_steps + 1 : end), nRun, pVars_test(trl_num), MSE_over_steps_test, axNum)
            end
        end
        
        if plotStatus
            if ismember(trl_num, floor(linspace(1, nTrlsTest, 6)))
                set(0, 'currentfigure', f2);
                print('-dtiff', '-r400', [rnnSubDir, RNNname, '_test_trl', num2str(iTrlID), '_num', num2str(trl_num), '_runs'])
                set(0, 'currentfigure', f);
                fTitle = ['[', RNNname, ']: test trial ID ', num2str(iTrlID), ' (#', num2str(trl_num), ')'];
                plot_costa_RNN_progress(f, nUnits, targets(:, :), R_test(:, iMdlSample_test), tRNN(:, iMdlSample_test), tData(n_pred_steps + 1 : end), nRun, pVars_test(trl_num, nRun), MSE_over_steps_test, trainRNN, fTitle)
                fAxs = findall(f, 'type', 'axes'); set(fAxs(2), 'ylim', [-0.2 1]);
                print('-dtiff', '-r400', [rnnSubDir, RNNname, '_test_trl', num2str(iTrlID), '_num', num2str(trl_num), '_overall'])
            end
        end
        
        % package up and save outputs
        RNN = make_RNN_struct(trainRNN, doSmooth, smoothWidth, meanSubtract, doSoftNorm, normByRegion, rmvOutliers, ...
            outliers, rnnDir, RNNname, dtFactor, g, alpha, tauRNN, tauWN, ampInWN, ampWN, nRunTrain, nonlinearity, ...
            nUnits, nTrls, nSets, arrayUnit, arrayRgns, iTrlID, trl_num, setID, R_test(:, iMdlSample_test), [], [], [], ...
            dtRNN, dtData, J_test, [], [], [], sum_MSE_over_runs_test, mean_MSE_over_runs_test, pVars_test, stdData_test, iPredict);
        
        if saveMdl
            save([rnnSubDir, RNNname, '_test_trl', num2str(iTrlID), '_num', num2str(trl_num) '.mat'], 'RNN', '-v7.3')
        end
        
        clear RNN
        trl_num = trl_num + 1;
        close all;
        toc
    end
    
    % get convergence metrics so you can compare train/test
    avg_final_pVar_test = mean(pVars_test(:, end));
    avg_final_mean_MSE_test = mean(mean_MSE_over_runs_test);
    avg_final_sum_MSE_test = mean(sum_MSE_over_runs_test);
end

end






