function RNN = fit_costa_RNN_prediction(RNNname, ...
    allSpikes, allPossTS, allEvents, arrayUnit, T, arrayRgns, ntwk_flavor, params)
% so many things at this point
% CS 2023
rng(42)
%% data meta
mouseVer        = '';
%% data parameters
dtData                  = 0.010;                % time step (in s) of the training data
dtFactor                = 20;                   % number of interpolation steps for RNN
doSmooth                = true;
smoothWidth             = 0.15;                 % in seconds, width of gaussian kernel if doSmooth == true
rmvOutliers             = true;
meanSubtract            = true;
doSoftNorm              = true;
normByRegion            = false;                % normalize activity by region or globally
%% RNN parameters
g                       = 1.5;                  % instability (chaos); g<1=damped, g>1=chaotic
tauRNN                  = 0.001;                % decay costant of RNN units ?in msec
tauWN                   = 0.1;                  % decay constant on filtered white noise inputs
ampInWN                 = 0.001;                % input amplitude of filtered white noise
%% training params
alpha                   = 1;                    % overall learning rate for regularizer
nonlinearity            = @tanh;                % inline function for nonlinearity
nRunTrain               = 1000;
nRunTest                = 1;                    % TO DO: IDK HOW MUCH TO TEST?!?!
trainFromPrev           = false;                % assume you're starting from beginning, but if not, feed in previous J
trainAllUnits           = true;                 % modify all units in network by default
trainRNN                = true;                 % true by default until it's not!
nTrlsIncluded           = 400;
nUnitsIncluded          = 25;
n_pred_steps            = 1;                    % each timestep = dtData (in this case currently 10ms)
use_reservoir           = 0;                    % default no. otherwise, make replicates so you have more units than targes
nReplicates             = [];                   % default no. if use_reservoir = true you gotta specify this
use_synthetic_targets   = false;                % default no. instead of real data, fake data
add_readout_unit        = false;
add_white_noise         = false; 
%% output options
plotStatus              = true;
saveMdl                 = true;
nBins                   = 50;
%% directory setup + final param setup

if exist('params', 'var')
    assignParams(who, params);
end

rnnDir          = ['~/Dropbox (BrAINY Crew)/costa_learning/models/', mouseVer, filesep]; % define output directories
rnnSubDir       = [rnnDir, RNNname(strfind(RNNname, '_') + 1 : end), filesep];
if ~isfolder(rnnSubDir)
    mkdir(rnnSubDir)
end
if ~isfolder([rnnDir, 'train_test_lists'])
    mkdir([rnnDir, 'train_test_lists'])
end
if ~isfolder([rnnDir 'exp_data' filesep]) 
    mkdir([rnnDir 'exp_data' filesep])
end

% generate subdir for this model flavor via a few identifiers
mdl_date = datetime; mdl_date.Format = 'yyyy-MM-dd'; mdl_date = char(mdl_date); mdl_date(strfind(mdl_date, '-')) = '_';
mdl_version_dir = [mdl_date, '_', ntwk_flavor];
if n_pred_steps > 1, mdl_version_dir = [mdl_version_dir, num2str(n_pred_steps)]; end
if use_reservoir, mdl_version_dir = [ mdl_version_dir, '_reservoir']; end
if use_synthetic_targets, mdl_version_dir = [mdl_version_dir, '_synthetic']; end
if add_white_noise, mdl_version_dir = [mdl_version_dir, '_noisy']; end
if add_readout_unit, mdl_version_dir = [mdl_version_dir, '_plusreadout']; end

mdl_version_dir = [mdl_version_dir filesep];
if ~isfolder([rnnSubDir, mdl_version_dir])
    mkdir([rnnSubDir, mdl_version_dir])
end

rnnSubDir = [rnnSubDir, mdl_version_dir];

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
[train_trl_IDs, test_trl_IDs, nTrlsTrain, nTrlsTest, start_trl_num, prevJ, prev_w_out, trainRNN, iPredict] = get_train_test_lists_and_progress(rnnDir, rnnSubDir, RNNname, nTrlsIncluded, nUnits, trainAllUnits, nUnitsIncluded);
nPredict = length(iPredict);
iNonTarget = find(~ismember(1 : nUnits, iPredict)); % sanity: assert(isempty(intersect(iNonTarget, iPredict)))

%% snipping out most units except those whose output we wanna predict
exp_data = exp_data(iPredict, :); % iPredict = units whose output we wanna predict, iTarget = units whose synapses we modify in trainiing

% update region indexing for use out of function (in case iPredict ~= 1 : nUnits)
linear_rgn_inds = arrayfun( @(iRgn) find(arrayRgns{iRgn, 3}), 1 : size(arrayRgns, 1), 'un', 0)';
linear_inds_included_units = arrayfun(@(iRgn) linear_rgn_inds{iRgn}(ismember(linear_rgn_inds{iRgn}, iPredict)), 1 : size(arrayRgns, 1), 'un', 0)';
arrayRgns = [arrayRgns(:, 1 : 2), linear_inds_included_units];
arrayUnit = arrayUnit(iPredict);

% if you're using replicates of real unit activities so as to have nUnits > #patterns
if use_reservoir
    exp_data = cell2mat(arrayfun(@(iUnit) repmat(exp_data(iUnit, :), nReplicates, 1), 1 : length(iPredict), 'un', 0)');
    nPredict = nPredict * nReplicates;
    iPlot = 1 : nReplicates : nPredict;
else
    iPlot = 1 : nPredict;
end
% update in-function indexing 
nUnits = size(exp_data, 1); % sloppy overwrite
iPredict = find(any(exp_data, 2));

%% cut down variable size for memory saving
if use_synthetic_targets
        % grab full set
        training_data = arrayfun(@(iTrlID) exp_data(:, fixOnInds(iTrlID) : fixOnInds(iTrlID + 1) - 1), train_trl_IDs(1 : end), 'un', 0)';
        testing_data = arrayfun(@(iTrlID) exp_data(:, fixOnInds(iTrlID) : fixOnInds(iTrlID + 1) - 1), test_trl_IDs(1 : end), 'un', 0)';
        all_data = [training_data; testing_data];
        % make the fakes
        synthetic_data  = generate_synthetic_data(all_data, 1 : nReplicates : nUnits, 1, 0.1, 1/4, smoothWidth, dtData);
        % split the fakes into inputs and targets separated by n_pred_steps
        synthetic_inputs = arrayfun(@(iTrlNum) synthetic_data{iTrlNum}(:, 1 : end - n_pred_steps), 1 : (nTrlsTrain + nTrlsTest), 'un', 0)';
        synthetic_targets = arrayfun(@(iTrlNum) synthetic_data{iTrlNum}(:, 1 + n_pred_steps : end), 1 : (nTrlsTrain + nTrlsTest), 'un', 0)';
        % split back into train and test again
        training_inputs = synthetic_inputs(1 : nTrlsTrain);             training_targets = synthetic_targets(1 : nTrlsTrain);
        testing_inputs = synthetic_inputs(nTrlsTrain + 1 : end);        testing_targets = synthetic_targets(nTrlsTrain + 1 : end);
else % DEFAULT
    training_inputs = arrayfun(@(iTrlID) exp_data(:, fixOnInds(iTrlID) : fixOnInds(iTrlID + 1) - 1 - n_pred_steps), train_trl_IDs(1 : end), 'un', 0)';
    training_targets = arrayfun(@(iTrlID) exp_data(:, fixOnInds(iTrlID) + n_pred_steps : fixOnInds(iTrlID + 1) - 1), train_trl_IDs(1 : end), 'un', 0)';
    testing_inputs = arrayfun(@(iTrlID) exp_data(:, fixOnInds(iTrlID) : fixOnInds(iTrlID + 1) - 1 - n_pred_steps), test_trl_IDs(1 : end), 'un', 0)';
    testing_targets = arrayfun(@(iTrlID) exp_data(:, fixOnInds(iTrlID) + n_pred_steps : fixOnInds(iTrlID + 1) - 1), test_trl_IDs(1 : end), 'un', 0)';
end

training_tData = arrayfun(@(iTrlID) allPossTS(fixOnInds(iTrlID) : fixOnInds(iTrlID + 1) - 1), train_trl_IDs(1 : end), 'un', 0)';
testing_tData = arrayfun(@(iTrlID) allPossTS(fixOnInds(iTrlID) : fixOnInds(iTrlID + 1) - 1), test_trl_IDs(1 : end), 'un', 0)';

% get thresholds for histogram
tmp_vals = [cell2mat(training_targets'), cell2mat(testing_targets')];
[~, lower_FR_thresh, upper_FR_thresh] = isoutlier(tmp_vals(:), 'percentiles', [0.5 99.5]);
clearvars exp_data allPossTS all_data synthetic_data synthetic_inputs synthetic_targets training_data testing_data tmp_vals fixOnInds

% set up labels for plots
paramText_train = sprintf(['TRAINING\nkernel = \t', num2str(smoothWidth), '\nn_t_r_l_s = \t', num2str(length([train_trl_IDs, test_trl_IDs])), '\ng = \t', num2str(g), '\nP_0 = \t', num2str(alpha), '\ntau_R_N_N = \t', num2str(tauRNN), '\ntau_W_N = \t', num2str(tauWN), '\nw_W_N = \t', num2str(ampInWN), '\nn_i_t_e_r = \t', num2str(nRunTrain), '\ndt_D_a_t_a = \t', num2str(dtData), '\ndt_R_N_N = \t', num2str(dtRNN)  ]); % this at the end!
paramText_test = sprintf(['TESTING\nkernel = \t', num2str(smoothWidth), '\nn_t_r_l_s = \t', num2str(length([train_trl_IDs, test_trl_IDs])), '\ng = \t', num2str(g), '\nP_0 = \t', num2str(alpha), '\ntau_R_N_N = \t', num2str(tauRNN), '\ntau_W_N = \t', num2str(tauWN), '\nw_W_N = \t', num2str(ampInWN), '\nn_i_t_e_r = \t', num2str(nRunTrain), '\ndt_D_a_t_a = \t', num2str(dtData), '\ndt_R_N_N = \t', num2str(dtRNN)  ]);

%% reading for training loop!
if trainRNN % TRAIN
    clearvars testing_inputs testing_targets testing_tData
    trl_num = start_trl_num;
    for iTrlID = train_trl_IDs(start_trl_num : end)
        fprintf('\n')
        fprintf([RNNname, ': training trial ', num2str(iTrlID), ', #=', num2str(trl_num), '...\n']), tic
            
        inputs = training_inputs{trl_num}; 
        switch ntwk_flavor
            case 'descriptive'
                targets = inputs;
            case {'1step', 'generative'}  
                targets = training_targets{trl_num};
        end
        
        tData = training_tData{trl_num}; nsp_Data = length(tData); 
        tRNN = tData(1) : dtRNN : tData(end); nsp_RNN = length(tRNN);
        
        % initialize weight matrix
        if trainFromPrev && iTrlID == train_trl_IDs(start_trl_num) && exist('prevJ', 'var') && ~isempty(prevJ)
            fprintf('Using prevJ from last trained trial. ')
            J = prevJ;
            w_out = prev_w_out;
        else
            if iTrlID == train_trl_IDs(1)
                J = g * randn(nUnits, nUnits) / sqrt(nUnits);
                w_out = (randn(nUnits, 1)) / sqrt(nUnits); % output weights to the readout unit
            end
        end
        
        % save initial J
        J0 = J;
        stdData = std(reshape(targets(iPredict, :), numel(targets(iPredict, :)), 1));
        
        % get indices for each sample of model data
        iMdlSample = zeros(nsp_Data - n_pred_steps, 1);
        for i = 1 : nsp_Data - n_pred_steps
            [~, iMdlSample(i)] = min(abs(tData(i) - tRNN));
        end
        
        all_coinciding_t = iMdlSample + dtFactor;
        
        % get indices for feedback term if generative
        feedback_indices = [1, 2 : dtFactor: (nsp_RNN - ((n_pred_steps - 1) * dtFactor) - (2 * dtFactor) + 1)]'; % TO DO TEST W N PRED STEPS > 1! either way should be indexed with iLearn, which goes max to nsp_Data - n_pred_steps
        
        % set up white noise inputs (from CURBD)
        if add_white_noise
            WN = get_frozen_input_WN(nUnits, ampWN, tauWN, ampInWN, nsp_RNN, dtRNN);
        end
        
        % initialize
        R = zeros(nUnits, nsp_RNN); % rate matrix - firing rates of neurons
        sum_MSE_over_runs = NaN(1, nRunTrain); mean_MSE_over_runs = NaN(1, nRunTrain); pVars = zeros(1, nRunTrain); cross_entropy = zeros(1, nRunTrain); 
        JR = zeros(nUnits, nsp_RNN); % z(t) for the output readout unit
        PJ = alpha * eye(nUnits); % learning update matrix (see Sussillo and Abbot, 2009); dim are pN x pN where p=fraction of neurons to modify - here it's all of them
        z_out = zeros(1, nsp_RNN); % readout unit
        % currently arbitrary readout target definition
        f_readout = rescale(median(targets, 1), -1, 1); f_readout_bin = double(f_readout >= 0); 
        
        % set up for histogram if you're plotting
        max_count = nsp_Data - n_pred_steps;
        if plotStatus && ismember(trl_num, floor(linspace(1, nTrlsTrain, 4))) 
            f = figure('color', 'w', 'Position', [100 100 1900 750]); axNum = 0; f2 = figure('color', 'w', 'Position', [100 100 1900 500]);
            axs = arrayfun( @(i) subplot(1, 4, i, 'NextPlot', 'add', 'box', 'off', 'tickdir', 'out', 'fontsize', 12, 'fontweight', 'bold'), 1 : 4);
            text(axs(4), 0.25 * range(get(axs(4), 'xlim')), 0.75 * range(get(axs(4), 'ylim')), paramText_train, 'fontsize', 14, 'fontweight', 'bold', 'horizontalalignment', 'center')
            set(axs(4), 'xtick', '', 'ytick', '')
        end
        
        %% loop through training runs
        for nRun = 1 : nRunTrain
            H = inputs(:, 1); % set initial condition
            R(:, 1) = tanh(H); % convert to currents through nonlinearity
            z_out(1) = w_out' * R(:, 1); % zi(t) = sum (Jij rj) over j
            % tLearn = 0;
            iLearn = 1; % keeps track of current time and last datapoint learned respectively
            MSE_over_steps = zeros(1, nsp_Data - n_pred_steps);
            for t = 2 : nsp_RNN - (n_pred_steps - 1) * dtFactor % 2 : nsp_RNN usually. this range will be identical for n_pred_steps < 2 as before
                % tLearn = tLearn + dtRNN; 
                R(:, t) = tanh(H); % generate output / compute next RNN step
                z_out(t) = w_out' * R(:, t);
                
%                 switch ntwk_flavor
%                     case 'descriptive'
%                         variable_term = zeros(nUnits, 1);
%                     case '1step'
%                         variable_term = inputs(:, iLearn);
%                     case 'generative'
%                         variable_term = R(:, feedback_indices(iLearn)); % feedback_indices == iMdlSample(iLearn - 1) + 1, except at iLearn = 1 (then it should = 1)
%                 end
                if isequal(ntwk_flavor, 'descriptive')
                    variable_term = zeros(nUnits, 1);
                elseif isequal(ntwk_flavor, '1step')
                    variable_term = inputs(:, iLearn);
                elseif isequal(ntwk_flavor, 'generative')
                    variable_term = R(:, feedback_indices(iLearn)); % feedback_indices == iMdlSample(iLearn - 1) + 1, except at iLearn = 1 (then it should = 1)
                end
                
                if add_white_noise
                    variable_term = variable_term + WN(:, t);
                end
                
                JR(:, t) = J * R(:, t) + variable_term;
                H = H + dtRNN * (-H + JR(:, t)) / tauRNN; % "hidden" activity updated w the update term, p much equivalent to: H + (dtRNN / tauRNN) * (-H + JR(:, tt));
                if t == all_coinciding_t(iLearn) % iMdlSample(iLearn) + dtFactor % tLearn >= dtData % update J if the RNN time coincides with a data point
                    % tLearn = 0;
                    % assert(t == iMdlSample(iLearn) + dtFactor)
                    error = R(:, t) - targets(:, iLearn);
                    readout_error = z_out(t) - f_readout(iLearn);
                    % readout_error = get_cross_entropy(f_readout_bin(iLearn), sigmoid(z_out(t)));
                    MSE_over_steps(iLearn) = mean(error .^ 2);
                    iLearn = iLearn + 1; % update learning index
                    if (nRun <= nRunTrain)
                        k = PJ * R(:, t); % N x 1 
                        rPr = R(:, t)' * k; % 1 x 1; inv xcorr of ntwk rates (when you square something it magnifies the sensitivity to changes)
                        c = 1 / (1 + rPr); % tune learning rate by looking at magnitude of model activity. if big changes in FRs and wanna move quickly/take big steps (aka momentum). as get closer, don't wanna overshoot, so take smaller steps
                        PJ = PJ - c * (k * k'); % use squared firing rates (R * R^T) to update PJ - maybe momentum effect?
                        J(:, :) = J(:, :) - c * error * k'; % update J (pre-syn wts adjusted according to post-syn target)
                        if add_readout_unit
                            w_out = w_out - (c * readout_error * k')'; % adjust reaodut weights using same c and P as for J
                        end
                    end
                end
            end
            
            % get evaluation metrics over runs
            rModelSample = R(iPredict, iMdlSample);
            pVar = 1 - ( norm(targets(iPredict, :) - rModelSample, 'fro' ) / ( sqrt(nPredict * (nsp_Data - n_pred_steps)) * stdData) ).^2;
            pVars(nRun) = pVar; mean_MSE_over_runs(nRun) = mean(MSE_over_steps); sum_MSE_over_runs(nRun) = sum(MSE_over_steps);
            cross_entropy(nRun) = readout_error(end);
            if ismember(nRun, [1 250 nRunTrain]) && ismember(trl_num, floor(linspace(1, nTrlsTrain, 4)))
                axNum = axNum + 1;
                plot_costa_RNN_param_comparisons(f2, axs, [RNNname(5 : end), ' (ID: ', num2str(iTrlID), ' / # ', num2str(trl_num), '):'], nPredict, targets(iPredict, :), R(iPredict, iMdlSample), tRNN(:, iMdlSample), tData(n_pred_steps + 1 : end), nRun, pVars, MSE_over_steps, axNum)
            end
        end
        if plotStatus
            if ismember(trl_num, floor(linspace(1, nTrlsTrain, 4)))
                progCompFigName = [rnnSubDir, RNNname, '_train', num2str(iTrlID), '_num', num2str(trl_num), '_runs'];
                overallCompFigName = [rnnSubDir, RNNname, '_train', num2str(iTrlID), '_num', num2str(trl_num), '_overall'];
                histName = [rnnSubDir, RNNname, '_train', num2str(iTrlID), '_num', num2str(trl_num), '_FR_hists'];
                set(0, 'currentfigure', f2); print('-dtiff', '-r400', progCompFigName)
                set(0, 'currentfigure', f); fTitle = ['[', RNNname, ']: train trial ID ', num2str(iTrlID), ' (#', num2str(trl_num), ')'];
                plot_costa_RNN_progress(f, nUnits, targets(:, :), R(:, iMdlSample), tRNN(:, iMdlSample), tData(n_pred_steps + 1 : end), nRun, pVars, sum_MSE_over_runs, trainRNN, '')
                fAxs = findall(f, 'type', 'axes'); set(fAxs(2), 'ylim', [-0.2 1]); print('-dtiff', '-r400', overallCompFigName)
                plot_comparative_FR_hists_each_unit(nBins, nUnits, nPredict, max_count, lower_FR_thresh, upper_FR_thresh, R(iPlot, iMdlSample), targets(iPlot, :), fTitle, use_reservoir); print('-dtiff', '-r400', histName)
                plot_comparative_FR_hists_pop(nBins, nUnits, nPredict, max_count, 0, 1, R(:, iMdlSample), targets, fTitle); print('-dtiff', '-r400', [histName, '_pop'])
            end
        end
        
        % get evaluation metrics after all runs
        X = targets(iPredict, :); Y = rModelSample;
        MAE = mean(abs(X(:) - Y(:))); MSE = mean((X(:) - Y(:)) .^ 2); RMSE = sqrt(MSE);
        
        % package up and save outputs
        [RNN] = make_RNN_struct(trainRNN, doSmooth, smoothWidth, meanSubtract, doSoftNorm, normByRegion, rmvOutliers, ...
            outliers, rnnDir, RNNname, dtFactor, g, alpha, tauRNN, tauWN, ampInWN, ampWN, nRunTrain, nonlinearity, ...
            train_trl_IDs, test_trl_IDs, nUnits, nTrls, nSets, arrayUnit, arrayRgns, iTrlID, trl_num, setID, R(:, iMdlSample), [], [], [], ...
            dtRNN, dtData, w_out, z_out, J, J0, [], [], sum_MSE_over_runs, mean_MSE_over_runs, pVars, [], ...
            iPredict, ntwk_flavor, n_pred_steps, use_reservoir, use_synthetic_targets, add_readout_unit, add_white_noise, trainAllUnits, nTrlsIncluded, ...
            MAE, MSE, RMSE, f_readout_bin, []);
        if saveMdl
            save([rnnSubDir, RNNname, '_train', num2str(iTrlID), '_num', num2str(trl_num) '.mat'], 'RNN', '-v7.3')
        end
        clear RNN, trl_num = trl_num + 1; close all; toc % housekeeping
    end
else % TEST
    start_trl_num = 1; trl_num = start_trl_num;
    if ~exist('J', 'var') % at present you run testing and training separately (not in same script call)
        fprintf('Using J from last trained trial.')
        J_test = prevJ;
        w_out_test = prev_w_out;
    else
        keyboard % just to catch potential J inconsistencies
    end
    
    pVars_test = zeros(nTrlsTest, nRunTest); sum_MSE_over_runs_test = zeros(nTrlsTest, nRunTest); mean_MSE_over_runs_test = zeros(nTrlsTest, nRunTest);
    for iTrlID = test_trl_IDs(start_trl_num : end)
        fprintf('\n'), fprintf([RNNname, ': testing trial ', num2str(iTrlID), ', #=', num2str(trl_num), '...']), tic
        
        inputs = testing_inputs{trl_num};
        switch ntwk_flavor
            case 'descriptive'
                targets = inputs;
            case{'1step', 'generative'}
                targets = testing_targets{trl_num}; 
        end
        tData = testing_tData{trl_num}; nsp_Data = length(tData); 
        tRNN = tData(1) : dtRNN : tData(end); nsp_RNN = length(tRNN);
        stdData_test = std(reshape(targets(iPredict, :), numel(targets(iPredict, :)), 1));
        iMdlSample_test = zeros(nsp_Data - n_pred_steps, 1);
        for i = 1 : nsp_Data - n_pred_steps
            [~, iMdlSample_test(i)] = min(abs(tData(i) - tRNN));
        end
        all_coinciding_t = iMdlSample_test + dtFactor;

        feedback_indices = [1, 2 : dtFactor: (nsp_RNN - ((n_pred_steps - 1) * dtFactor) - (2 * dtFactor) + 1)]'; % TO DO TEST W N PRED STEPS > 1! either way should be indexed with iLearn, which goes max to nsp_Data - n_pred_steps
        
        if add_white_noise
            WN = get_frozen_input_WN(nUnits, ampWN, tauWN, ampInWN, nsp_RNN, dtRNN);
        end
        
        R_test = zeros(nUnits, nsp_RNN); JR_test = zeros(nUnits, nsp_RNN);
        
        % addition of a readout unit w kinda arbitrary targets!
        z_out_test = zeros(1, nsp_RNN);
        f_readout = rescale(median(targets, 1), -1, 1); f_readout_bin = double(f_readout >= 0);

        max_count = nsp_Data - n_pred_steps; % set up for histogram
        if plotStatus && ismember(trl_num, floor(linspace(1, nTrlsTest, 4))) % plot progress as we tweak parameters
            f = figure('color', 'w', 'Position', [100 100 1900 750]); f2 = figure('color', 'w', 'Position', [100 100 1900 500]);
            axs = arrayfun( @(i) subplot(1, 4, i, 'NextPlot', 'add', 'box', 'off', 'tickdir', 'out', 'fontsize', 12, 'fontweight', 'bold'), 1 : 4);
            text(axs(4), 0.25 * range(get(axs(4), 'xlim')), 0.75 * range(get(axs(4), 'ylim')), paramText_test, 'fontsize', 14, 'fontweight', 'bold', 'horizontalalignment', 'center')
            axNum = 0; set(axs(4), 'xtick', '', 'ytick', '')
        end
        
        %% loop through testing runs
        for nRun = 1 : nRunTest
            H = inputs(:, 1);
            R_test(:, 1) = tanh(H);
            z_out_test(1) = w_out_test' * R_test(:, 1);
            % tLearn = 0;
            iLearn = 1;
            MSE_over_steps_test = zeros(1, nsp_Data - n_pred_steps);
            for t = 2 : nsp_RNN - (n_pred_steps - 1) * dtFactor 
                % tLearn = tLearn + dtRNN;
                R_test(:, t) = tanh(H);
                z_out_test(t) = w_out_test' * R_test(:, t);
%                 switch ntwk_flavor
%                     case 'descriptive'
%                         variable_term = zeros(nUnits, 1);
%                     case '1step'
%                         variable_term = inputs(:, iLearn);
%                     case 'generative'
%                         variable_term = R_test(:, feedback_indices(iLearn));
%                 end

                if isequal(ntwk_flavor, 'descriptive')
                    variable_term = zeros(nUnits, 1);
                elseif isequal(ntwk_flavor, '1step')
                    variable_term = inputs(:, iLearn);
                elseif isequal(ntwk_flavor, 'generative')
                    variable_term = R_test(:, feedback_indices(iLearn)); % feedback_indices == iMdlSample(iLearn - 1) + 1, except at iLearn = 1 (then it should = 1)
                end
                
                if add_white_noise
                    variable_term = variable_term + WN(:, t);
                end
                JR_test(:, t) = J_test * R_test(:, t) + variable_term;
                H = H + dtRNN * (-H + JR_test(:, t)) / tauRNN;
                if t == all_coinciding_t(iLearn)% iMdlSample_test(iLearn) + dtFactor % tLearn >= dtData
                    % tLearn = 0;
                    error = R_test(:, t) - targets(:, iLearn);
                    readout_error = z_out_test(t) - f_readout(iLearn);
                    % readout_error = get_cross_entropy(f_readout_bin(iLearn), sigmoid(z_out_test(t))); 
                    MSE_over_steps_test(iLearn) = mean(error .^ 2);
                    iLearn = iLearn + 1;
                end
            end
            rModelSample_test = R_test(iPredict, iMdlSample_test);
            pVar = 1 - ( norm(targets(iPredict, :) - rModelSample_test, 'fro' ) / ( sqrt(nPredict * (nsp_Data - n_pred_steps)) * stdData_test) ) .^ 2;
            pVars_test(trl_num, nRun) = pVar; mean_MSE_over_runs_test(trl_num, nRun) = mean(MSE_over_steps_test); sum_MSE_over_runs_test(trl_num, nRun) = sum(MSE_over_steps_test);
            if ismember(nRun, [1 250 nRunTrain]) && ismember(trl_num, floor(linspace(1, nTrlsTest, 4)))
                axNum = axNum + 1;
                plot_costa_RNN_param_comparisons(f2, axs, [RNNname(5 : end), ' (ID: ', num2str(iTrlID), ' / # ', num2str(trl_num), '):'], nPredict, targets(iPredict, :), R_test(iPredict, iMdlSample_test), tRNN(:, iMdlSample_test), tData(n_pred_steps + 1 : end), nRun, pVars_test(trl_num), MSE_over_steps_test, axNum)
            end
        end
        if plotStatus
            if ismember(trl_num, floor(linspace(1, nTrlsTest, 4)))
                progCompFigName = [rnnSubDir, RNNname, '_test', num2str(iTrlID), '_num', num2str(trl_num), '_runs'];
                overallCompFigName = [rnnSubDir, RNNname, '_test', num2str(iTrlID), '_num', num2str(trl_num), '_overall'];
                histName = [rnnSubDir, RNNname, '_test', num2str(iTrlID), '_num', num2str(trl_num), '_FR_hists'];
                set(0, 'currentfigure', f2); print('-dtiff', '-r400', progCompFigName)
                set(0, 'currentfigure', f); fTitle = ['[', RNNname, ']: test trial ID ', num2str(iTrlID), ' (#', num2str(trl_num), ')'];
                plot_costa_RNN_progress(f, nUnits, targets(:, :), R_test(:, iMdlSample_test), tRNN(:, iMdlSample_test), tData(n_pred_steps + 1 : end), nRun, pVars_test(trl_num, nRun), MSE_over_steps_test, trainRNN, fTitle)
                fAxs = findall(f, 'type', 'axes'); set(fAxs(2), 'ylim', [-0.2 1]); print('-dtiff', '-r400', overallCompFigName)
                plot_comparative_FR_hists_each_unit(nBins, nUnits, nPredict, max_count, lower_FR_thresh, upper_FR_thresh, R_test(iPlot, iMdlSample_test), targets(iPlot, :), fTitle, use_reservoir); print('-dtiff', '-r400', histName)
                plot_comparative_FR_hists_pop(nBins, nUnits, nPredict, max_count, 0, 1, R_test(:, iMdlSample_test), targets, fTitle); print('-dtiff', '-r400', [histName, '_pop'])
            end
        end
        
        % get test evaluation metrics
        X = targets(iPredict, :); Y = rModelSample_test;
        MAE = mean(abs(X(:) - Y(:))); MSE = mean((X(:) - Y(:)) .^ 2); RMSE = sqrt(MSE);
        
        % package up and save outputs
        RNN = make_RNN_struct(trainRNN, doSmooth, smoothWidth, meanSubtract, doSoftNorm, normByRegion, rmvOutliers, ...
            outliers, rnnDir, RNNname, dtFactor, g, alpha, tauRNN, tauWN, ampInWN, ampWN, nRunTrain, nonlinearity, ...
            train_trl_IDs, test_trl_IDs, nUnits, nTrls, nSets, arrayUnit, arrayRgns, iTrlID, trl_num, setID, R_test(:, iMdlSample_test), [], [], [], ...
            dtRNN, dtData, w_out_test, z_out_test, J_test, [], [], [], sum_MSE_over_runs_test, mean_MSE_over_runs_test, pVars_test, [], ...
            iPredict, ntwk_flavor, n_pred_steps, use_reservoir, use_synthetic_targets, add_readout_unit, add_white_noise, trainAllUnits, nTrlsIncluded, ...
            MAE, MSE, RMSE, f_readout_bin, []);
        if saveMdl
            save([rnnSubDir, RNNname, '_test', num2str(iTrlID), '_num', num2str(trl_num) '.mat'], 'RNN', '-v7.3')
        end
        clear RNN, trl_num = trl_num + 1; close all; toc % housekeeping
    end
    
    % get convergence/perofrmance metrics over full set so you can compare train/test
    all_iPredicts = cell2mat(arrayfun(@(iTrl) getfield(load([rnnSubDir, dir([rnnSubDir, RNNname, ...
        '_train', num2str(train_trl_IDs(iTrl)), ...
        '_num', num2str(iTrl) '.mat']).name]), 'RNN', 'mdl', 'iPredict'), 1 : nTrlsTrain, 'un', 0)');
    assert(isequal(unique(all_iPredicts, 'rows'), iPredict)) % sanity check: all iPredicts for all trials in training set must be identical and equal to iPredict
    
    % get convergence metrics from trainso you can compare train/test
    all_pVars = cell2mat(arrayfun(@(iTrl) getfield(load([rnnSubDir, dir([rnnSubDir, RNNname, ...
        '_train', num2str(train_trl_IDs(iTrl)), ...
        '_num', num2str(iTrl) '.mat']).name]), 'RNN', 'mdl', 'pVars'), 1 : nTrlsTrain, 'un', 0)'); all_final_pVars = all_pVars(:, end);
    all_mean_MSE_over_runs = cell2mat(arrayfun(@(iTrl) getfield(load([rnnSubDir, dir([rnnSubDir, RNNname, ...
        '_train', num2str(train_trl_IDs(iTrl)), ...
        '_num', num2str(iTrl) '.mat']).name]), 'RNN', 'mdl', 'mean_MSE_over_runs'), 1 : nTrlsTrain, 'un', 0)'); all_final_mean_MSE = all_mean_MSE_over_runs(:, end);
    all_cum_MSE_over_runs = cell2mat(arrayfun(@(iTrl) getfield(load([rnnSubDir, dir([rnnSubDir, RNNname, ...
        '_train', num2str(train_trl_IDs(iTrl)), ...
        '_num', num2str(iTrl) '.mat']).name]), 'RNN', 'mdl', 'cumulative_MSE_over_runs'), 1 : nTrlsTrain, 'un', 0)'); all_final_cum_MSE = all_cum_MSE_over_runs(:, end);
    
    % train over trials
    avg_final_pVar_train = mean(all_final_pVars);
    avg_final_mean_MSE_train = mean(all_final_mean_MSE);
    avg_final_sum_MSE_train = mean(all_final_cum_MSE);
    
    % test over trials
    avg_final_pVar_test = mean(pVars_test(:, end));
    avg_final_mean_MSE_test = mean(mean_MSE_over_runs_test);
    avg_final_sum_MSE_test = mean(sum_MSE_over_runs_test);
    
    % combine all trials and calculate metrics from concatenated (vs metrics per trial and then taking average)
    pred_activity_train_cell = arrayfun(@(iTrl) getfield(load([rnnSubDir, dir([rnnSubDir, RNNname, ...
        '_train', num2str(train_trl_IDs(iTrl)), '_num', num2str(iTrl) '.mat']).name]), ...
        'RNN', 'mdl', 'RMdlSample'), 1 : nTrlsTrain, 'un', 0); pred_activity_train = cell2mat(pred_activity_train_cell);
    pred_activity_test_cell = arrayfun(@(iTrl) getfield(load([rnnSubDir, dir([rnnSubDir, RNNname, ...
        '_test', num2str(test_trl_IDs(iTrl)), '_num', num2str(iTrl) '.mat']).name]), ...
        'RNN', 'test', 'RMdlSample_test'), 1 : nTrlsTest, 'un', 0); pred_activity_test = cell2mat(pred_activity_test_cell);
    targ_activity_train = cell2mat(training_targets'); targ_activity_test = cell2mat(testing_targets');
    X_train = targ_activity_train; Y_train = pred_activity_train; X_test = targ_activity_test; Y_test = pred_activity_test;
    
    % plot for each unit its mean firing pattern over trials compared to model
    if ~use_synthetic_targets
        plot_mean_SEM_FR_each_unit(length(iPlot), pred_activity_train_cell', training_targets, RNNname, use_reservoir, nReplicates)
        print('-dtiff', '-r400', [rnnSubDir, RNNname, '_ mean_firing_patterns_train']), close
    end
    plot_mean_SEM_FR_each_unit(length(iPlot), pred_activity_test_cell', testing_targets, RNNname, use_reservoir, nReplicates)
    print('-dtiff', '-r400', [rnnSubDir, RNNname, '_ mean_firing_patterns_test']), close
    
    % global metrics
    MAE_train = mean(abs(X_train(:) - Y_train(:))); MAE_test = mean(abs(X_test(:) - Y_test(:)));
    MSE_train = mean((X_train(:) - Y_train(:)) .^ 2); MSE_test = mean((X_test(:) - Y_test(:)) .^ 2);
    RMSE_train = sqrt(MSE_train); RMSE_test = sqrt(MSE_test);
    pVar_train = 1 - ( norm(X_train - Y_train, 'fro' ) / ( sqrt(numel(X_train)) * std(reshape(X_train, numel(X_train), 1))) ) .^ 2;
    pVar_test = 1 - ( norm(X_test - Y_test, 'fro' ) / ( sqrt(numel(X_test)) * std(reshape(X_test, numel(X_test), 1))) ) .^ 2;
    
    % save overall params and metrics for full train/test set
    if saveMdl
        [~, currentParams] = make_RNN_struct(trainRNN, doSmooth, smoothWidth, meanSubtract, doSoftNorm, normByRegion, rmvOutliers, ...
            outliers, rnnDir, RNNname, dtFactor, g, alpha, tauRNN, tauWN, ampInWN, ampWN, nRunTrain, nonlinearity, ...
            train_trl_IDs, test_trl_IDs, nUnits, nTrls, nSets, arrayUnit, arrayRgns, iTrlID, nTrlsTest, setID, R_test(:, iMdlSample_test), [], [], [], ...
            dtRNN, dtData, [], [], [], [], [], [], sum_MSE_over_runs_test, mean_MSE_over_runs_test, pVars_test, [], ...
            iPredict, ntwk_flavor, n_pred_steps, use_reservoir, use_synthetic_targets, add_readout_unit, add_white_noise, trainAllUnits, nTrlsIncluded, ...
            MAE, MSE, RMSE, [], []);
        save([rnnSubDir, RNNname, '_metrics.mat'], 'currentParams', ...
            'avg_final_pVar_train', 'avg_final_mean_MSE_train', 'avg_final_sum_MSE_train', ...
            'avg_final_pVar_test', 'avg_final_mean_MSE_test', 'avg_final_sum_MSE_test', ...
            'pVar_train', 'pVar_test', 'MAE_train', 'MAE_test', 'MSE_train', 'MSE_test', 'RMSE_train', 'RMSE_test', ...
            '-v7.3')
    end
end

end






