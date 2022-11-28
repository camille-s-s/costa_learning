clearvars;
close all

%% params

nSamples        = 21; % for within one trial
mdl_version     = '1step_prediction'; %'v3test'; % ''; % 'v3test/'; % or ''

%% in and outdirs

bd              = '~/Dropbox (BrAINY Crew)/costa_learning/';
mdlDir          = [bd 'models/', mdl_version, filesep];
RNNfigdir       = [bd 'figures/', mdl_version, filesep];
RNNSampleDir    = [mdlDir 'model_samples/'];
spikeInfoPath   = [bd 'reformatted_data/'];

%%

addpath(genpath(bd))

if ~isfolder(RNNSampleDir)
    mkdir(RNNSampleDir)
end

cd(spikeInfoPath)

allFiles = dir('*_meta.mat');
bad_files = arrayfun(@(iFile) any(strfind(allFiles(iFile).name, '._')), 1 : length(allFiles));
allFiles(bad_files) = [];

if strcmp (computer, 'MACI64')
    fontSzS = 10;
    fontSzM = 14;
    fontSzL = 16;
elseif strcmp(computer, 'GLNXA64')
    fontSzS = 6;
    fontSzM = 8;
    fontSzL = 9;
end

for iFile = 1 : length(allFiles)
    
    fName = allFiles(iFile).name;
    fID = fName(1 : strfind(fName, '_') - 1);
    monkey = fID(1);
    ssnDate = fID(2 : end);
    cd([mdlDir, monkey, ssnDate, filesep])
    currSsn = dir('rnn_*_train_trl*_num*.mat');
    figNameStem = [monkey, ssnDate, '_', mdl_version];
    allTrialIDs = unique(arrayfun(@(i) ...
        str2double(currSsn(i).name(strfind(currSsn(i).name, '_trl') + 4 : strfind(currSsn(i).name, '_num') - 1)), 1 : length(currSsn)));
    
    % for first trial of a session, pull params (this field is empty for all other trials in a session)
    % TO DO: ADD SOME KIND OF PARAM CHECK THAT THE PARAMS IN RNN.PARAMS (*_num1.mat) ARE WHAT YOU THINK THEY ARE!
    paramsRNN = load(dir('rnn_*_train_trl*_num1.mat').name);
    params = paramsRNN.RNN.params;
    dtData = params.dtData; % in sec
    dtRNN = params.dtRNN;
    
    M = load([spikeInfoPath, monkey, ssnDate, '_meta.mat']);
    trlInfo = M.trlInfo;
    allPossTS = M.dsSpikeTimes;
    % TO DO: GET TDATA AND TRNN FROM ALLPOSSTS
    % TO DO: write sanity check from load(RNN.params.exp_data_path) for exp_data vs allSpikes from costa_RNN_*_wrapper (run through preprocess_data_for_RNN_trainin)
    % get a bunch of useful event variables as well as raw data
    load([mdlDir, 'exp_data', filesep, 'rnn_', monkey, ssnDate, '_exp_data.mat'], 'exp_data', 'outliers', 'fixOnInds', 'stimTimeInds', 'nTrls', 'nTrlsPerSet', 'nSets', 'setID', 'arrayUnit', 'arrayRgns');
    nTrlsInSession = nTrls; clear nTrls
    nTrlsAvailable = length(currSsn);
    % nTrlsIncluded = 32; % **********
    
    % get train/test list info
    load([mdlDir, 'train_test_lists' filesep, 'rnn_', monkey, ssnDate, '_train_test_list.mat'], 'train_trl_IDs', 'test_trl_IDs')
    all_trl_IDs = [train_trl_IDs, test_trl_IDs];
    
    % load all inputs, all targets, tData, and tRNN
    all_inputs = cell(nTrlsAvailable, 1);
    all_targets = cell(nTrlsAvailable, 1);
    all_tData = cell(nTrlsAvailable, 1);
    all_tRNN = cell(nTrlsAvailable, 1);
    trl_num = 1;
    
    for iTrlID = all_trl_IDs(1 : nTrlsAvailable)
        iStart = fixOnInds(iTrlID);
        iStop = fixOnInds(iTrlID + 1) - 1;
        all_inputs{trl_num} = exp_data(:, iStart : iStop - 1);
        all_targets{trl_num}  = exp_data(:, iStart + 1 : iStop);
        all_tData{trl_num}  = allPossTS(iStart : iStop);
        all_tRNN{trl_num}  = all_tData{trl_num}(1) : dtRNN : all_tData{trl_num}(end);
        trl_num = trl_num + 1;
        
        % TO DO: MAKE SURE ALL_TRL_IDs are in same order as how you pull the rest of the data, or better yet, fix it to
        % be so!)
    end
    
    % generate iModelSample for indices in RNN time coinciding with data time
    alliModelSamples = cell(nTrlsAvailable, 1);
    for iTrl = 1 : nTrlsAvailable
        
        iModelSample = zeros(length(all_tData{iTrl}) - 1, 1); % get indices for each sample of model data
        
        for i = 1 : length(all_tData{iTrl}) - 1
            [~, iModelSample(i)] = min(abs(all_tData{iTrl}(i) - all_tRNN{iTrl}));
        end
        
        alliModelSamples{iTrl} = iModelSample;
    end
    
    % TO DO: write sanity check that shared variables in exp_data vs RNN.params(1) are identical!
    allSetIDs = setID(all_trl_IDs(1 : nTrlsAvailable));
    
    %% COLLECT DATA FROM ALL TRIALS FOR A GIVEN SESSION
    
    rgnColors       = [1 0 0; 1 0 0; 1 0 1; 1 0 1; 0 0 1; 0 0 1; 0 1 0; 0 1 0];
    rgnLinestyle    = {'-', '-.', '-', '-.', '-', '-.', '-', '-.'};
    JDim            = size(paramsRNN.RNN.mdl.J);
    
    % collect outputs for all trials in a session
    all_trl_nums        = NaN(1, nTrlsAvailable);
    R_U                 = cell(nTrlsAvailable, 1);
    R_U                 = cell(nTrlsAvailable, 1);
    J_U                 = NaN([JDim, nTrlsAvailable]);
    J_U                 = NaN([JDim, nTrlsAvailable]);
    J0_U                = NaN([JDim, nTrlsAvailable]);
    pVarsTrls           = zeros(nTrlsAvailable, 1);
    sum_MSE_trls        = zeros(nTrlsAvailable, 1);
    fittedJ             = cell(nTrlsAvailable, 1);
    intraRgnJSampleAll  = cell(nTrlsAvailable, 1);
    all_iTargets        = cell(nTrlsAvailable, 1);
    
    for i = 1 : nTrlsAvailable % TO DO: SHOULD ADD CONSTRAINT THAT nTRLSAVAILABLE GO TO (AT MAX) NTRAINTRIALS
        mdlfnm              = dir([mdlDir, monkey, ssnDate, filesep, 'rnn_', monkey, ssnDate, '_train_trl', num2str(train_trl_IDs(i)), '_num', num2str(i), '.mat']);
        tmp_mdl             = load(mdlfnm.name);
        mdl                 = tmp_mdl.RNN.mdl;
        all_trl_nums(i)     = mdl.trl_num;
        R_U{i}              = mdl.RMdlSample;
        J_U(:, :, i)        = mdl.J;
        J0_U(:, :, i)       = mdl.J0;
        pVarsTrls(i)        = mdl.pVars(end);
        sum_MSE_trls(i)     = mdl.cumulative_MSE_over_runs(end);
        all_iTargets{i}     = mdl.iTarget;
        
        % collect first and last to assess convergence
        if i == 1
            firstPVars      = mdl.pVars;
            firstMSE        = mdl.cumulative_MSE_over_runs;
        end
        if i == nTrlsAvailable
            lastPVars       = mdl.pVars;
            lastMSE         = mdl.cumulative_MSE_over_runs;
        end
    end
    
    % sanity check re iTargets (they all should be the same for the training/test set of a given session)
    assert(size(unique(cell2mat(all_iTargets), 'rows'), 1) == 1)
    iTarget = all_iTargets{1};
    nPlasticUnits = length(iTarget);
    
    %% HOUSEKEEPING: SPATIAL
    
    desiredOrder = {'left_cdLPFC', 'right_cdLPFC', ...
        'left_mdLPFC', 'right_mdLPFC', ...
        'left_vLPFC', 'right_vLPFC', ...
        'left_rdLPFC','right_rdLPFC'}; % **********
    min_n_units_to_plot = 5; % **********
    [rgns, badRgn, newOrder, nUnitsAll, JLblPos, JLinePos] = reorder_units_by_region(arrayRgns, desiredOrder, min_n_units_to_plot);
    nRegions = size(rgns, 1);
    
    % TO DO: MOVE THIS SOMEWHERE AND UNCOMMENT IT AND STUFF AND MAKE IT WORK
    % % first get a global measure of synchrony between units (pairwise pearson correlation coefficient for all units over the
    % % timepoints spanned by dev dataset)
    % [R, P, RL, RU] = corrcoef(all_inputs');
    %
    % figure('color', 'w', 'Position', [100 100 1200 9000]);
    % imagesc(R), axis square
    % cm = brewermap(250,'*RdBu');
    % colormap(cm), colorbar, set(gca, 'clim', [-1 1])
    % arrayfun(@(iRgn) line(gca, [JLinePos(iRgn) JLinePos(iRgn)], get(gca, 'ylim'), 'linewidth', 1.5, 'color', 'k'), 1 : length(JLinePos))
    % arrayfun(@(iRgn) line(gca, get(gca, 'xlim'), [JLinePos(iRgn) JLinePos(iRgn)], 'linewidth', 1.5, 'color', 'k'), 1 : length(JLinePos))
    % set(gca, 'xtick', JLblPos, 'xticklabel', rgns(:, 1))
    % set(gca, 'ytick', JLblPos, 'yticklabel', rgns(:, 1))
    % set(gca, 'fontweight', 'bold', 'fontsize', 12)
    % title('Pearson correlation coefficient for experimental data')
    %
    %
    % % only plot R for most significant correlations with most stringent bonferroni correction
    % nComparisons = 0.5 * (nUnits^2 - nUnits);
    % bonferroni_alpha = 0.05 / nComparisons;
    % R_thresh = R;
    % R_thresh(P > bonferroni_alpha) = 0;
    %
    % figure('color', 'w', 'Position', [100 100 1200 9000]);
    % imagesc(R_thresh), axis square
    % cm = brewermap(250,'*RdBu');
    % colormap(cm), colorbar, set(gca, 'clim', [-1 1])
    % arrayfun(@(iRgn) line(gca, [JLinePos(iRgn) JLinePos(iRgn)], get(gca, 'ylim'), 'linewidth', 1.5, 'color', 'k'), 1 : length(JLinePos))
    % arrayfun(@(iRgn) line(gca, get(gca, 'xlim'), [JLinePos(iRgn) JLinePos(iRgn)], 'linewidth', 1.5, 'color', 'k'), 1 : length(JLinePos))
    % set(gca, 'xtick', JLblPos, 'xticklabel', rgns(:, 1))
    % set(gca, 'ytick', JLblPos, 'yticklabel', rgns(:, 1))
    % set(gca, 'fontweight', 'bold', 'fontsize', 12)
    
    %% HOUSEKEEPING: TEMPORAL
    
    % make sure that initial J of one trial is the final J of the previous trial
    assert(all(arrayfun(@(t) isequal(J0_U(:, :, t + 1), J_U(:, :, t)), 1 : nTrlsAvailable - 1)))
    
    % sanity check that all J0s are different
    J0_resh = cell2mat(arrayfun(@(iTrl) ...
        reshape(squeeze(J0_U(:, :, iTrl)), 1, size(J0_U, 1)^2), 1 : size(J0_U, 3), 'un', 0)');
    assert(isequal(size(unique(J0_resh, 'rows'), 1), nTrlsAvailable))
    
    % compute some useful quantities. note that _U is in same unit order as spikeInfo
    trlDursRNN = cellfun(@length, all_tData); % arrayfun(@(iTrl) size(D_U{iTrl}, 2), 1 : nTrlsAvailable)';
    shortestTrl = min(trlDursRNN(:));
    
    %% ASSESS CONVERGENCE
    
    convergenceFigDir = [RNNfigdir, 'convergence', filesep]; % pVarOverTrls, first trl convergence, last trl convergence
    if ~isfolder(convergenceFigDir)
        mkdir(convergenceFigDir)
    end
    
    figure, set(gcf, 'units', 'normalized', 'outerposition', [0.1 0.25 0.8 0.5]),
    plot(pVarsTrls, 'linewidth', 1.5, 'color', [0.05 0.4 0.15]), set(gca, 'ylim', [0.7 1.1], 'xlim', [1 nTrlsAvailable]), grid minor
    set(gca, 'fontweight', 'bold')
    title(gca, [monkey, ssnDate, ': pVar over trials'], 'fontsize', fontSzM, 'fontweight', 'bold')
    xlabel('# trials', 'fontsize', fontSzM, 'fontweight', 'bold')
    print('-dtiff', '-r400', [convergenceFigDir, 'pVar_over_trls_', figNameStem])
    close
    
    % summary convergence plots for first trial in training
    f = figure('color', 'w', 'Position', [100 100 1900 750]);
    fTitle = ['[', monkey, ssnDate, ']: train trial ID ', num2str(all_trl_IDs(1)), ' (#', num2str(1), ')'];
    plot_costa_RNN_progress(f, nPlasticUnits, all_targets{1}(iTarget, :), R_U{1}, all_tRNN{1}(alliModelSamples{1}), all_tData{1}, params.nRunTrain, firstPVars, firstMSE, 1, fTitle)
    print('-dtiff', '-r400', [convergenceFigDir, 'first_trl_convergence_', figNameStem])
    close
    
     % summary convergence plots for last trial in training
    f = figure('color', 'w', 'Position', [100 100 1900 750]);
    fTitle = ['[', monkey, ssnDate, ']: train trial ID ', num2str(all_trl_IDs(nTrlsAvailable)), ' (#', num2str(nTrlsAvailable), ')'];
    plot_costa_RNN_progress(f, nPlasticUnits, all_targets{nTrlsAvailable}(iTarget, :), R_U{nTrlsAvailable}, all_tRNN{nTrlsAvailable}(alliModelSamples{nTrlsAvailable}), all_tData{nTrlsAvailable}, params.nRunTrain, lastPVars, lastMSE, 1, fTitle)
    print('-dtiff', '-r400', [convergenceFigDir, 'latest_trl_convergence_', figNameStem])
    close

    %% TO DO: CONTINUE FROM REORDER_UNITS_BY_REGION
    
%     newRgnInds = [0, JLinePos];
%     
%     % get region labels for CURBD
%     curbdRgns = [rgns(:, 1), arrayfun(@(i) newRgnInds(i) + 1 : newRgnInds(i + 1), 1 : nRegions, 'un', 0)'];
%     
%     % J is reordered by region, so R_U must be as well. also reorder actual activity
%     exp_data_reordered = exp_data(newOrder, :);
%     R_U_reordered = arrayfun(@(iTrl) R_U{iTrl}(newOrder, :), 1 : nTrlsAvailable, 'un', 0)';
%     J_U_reordered = J_U(newOrder, newOrder, :);
%     
%     % TO DO: reorder all_inputs and all_targets
%     
%     
%     %% reorder full J and use it to get intra/inter only Js + plot J, R distributions + CURBD
%     
%     histFigDir = [RNNfigdir, 'histograms', filesep]; % pVarOverTrls, first trl convergence, last trl convergence
%     if ~isfolder(histFigDir)
%         mkdir(histFigDir)
%     end
%     
%     CURBDFigDir = [RNNfigdir, 'CURBD', filesep]; % pVarOverTrls, first trl convergence, last trl convergence
%     if ~isfolder(CURBDFigDir)
%         mkdir(CURBDFigDir)
%     end
%     % TO DO: STOPPED HERE! FIRST THING 10/6/22!! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     % for pulling trials from within a fixed time duration
%     ts = trlInfo.event_times;
%     lastTrlWithinHr = find(ts(:, 4) <= ts(1, 1) + 7200, 1, 'last') - 1;
%     latestSet = setID(lastTrlWithinHr) - 1;
%     
%     % set up subsampling of J, R, CURBD over a fixed duration and number of trials (fixed across monkeys and
%     % recording days)
%     if latestSet <= setID(nTrlsAvailable)
%         setsToPlot = floor(linspace(2, latestSet, 20));
%     else
%         setsToPlot = floor(linspace(2, setID(nTrlsAvailable), 20));
%     end
%     
%     allTrlIDs = cell2mat(arrayfun(@(iSet) find(setID == iSet, minTrlsAvailablePerSet), setsToPlot, 'un', 0)');
%     J_allsets = J_U_reordered(:, :, allTrlIDs);
%     R_allsets_trunc_cell = arrayfun(@(iTrl) R_U_reordered{iTrl}(:, 1 : shortestTrl), allTrlIDs, 'un', 0);
%     R_allsets_trunc_mat = cat(3, R_allsets_trunc_cell{:});
%     
%     % define histlims for J and R
%     nBins = 50;
%     JAllHistMin = min(sqrt(nUnits) * J_allsets(:));
%     JAllHistMax = max(sqrt(nUnits) * J_allsets(:));
%     JAllHistEdges = linspace(floor(JAllHistMin), ceil(JAllHistMax), nBins + 1);
%     JAllHistCenters = JAllHistEdges(1 : end - 1) + (diff(JAllHistEdges) ./ 2); % get the centers. checked
%     reshaped_J_allsets_mean = reshape(sqrt(nUnits) * mean(J_allsets, 3), nUnits .^ 2, 1);
%     JAllHistCounts = histcounts(reshaped_J_allsets_mean, JAllHistEdges); % TO DO: look at it with hist and see
%     RXLims = [-0.015 0.35]; % prctile(R_allsets_trunc_mat(:), [1 99]);
%     RAllHistEdges = linspace(RXLims(1), RXLims(2),  nBins + 1);
%     RAllHistCenters = RAllHistEdges(1 : end - 1) + (diff(RAllHistEdges) ./ 2);
%     RAllHistCounts = histcounts(mean(R_allsets_trunc_mat, 3), RAllHistEdges);
%     % TRIAL-AVERAGED OVER ALL SUBSAMPLED TRIALS: J and R HISTS
%     figure('color', 'w');
%     set(gcf, 'units', 'normalized', 'outerposition', [0.2 0 0.4 1])
%     subplot(2, 1, 1),
%     semilogy(JAllHistCenters, JAllHistCounts ./ sum(JAllHistCounts, 2), 'o-', 'linewidth', 1.25, 'markersize', 5);
%     set(gca, 'fontsize', fontSzM, 'fontweight', 'bold', 'ylim', [0 1], 'xlim', [-40 40]),
%     xlabel(gca, 'weight'), ylabel(gca, 'log density')
%     title(['[', monkey, ssnDate, '] trial-averaged J over subsampled trials'], 'fontweight', 'bold')
%     subplot(2, 1, 2),
%     semilogy(RAllHistCenters, RAllHistCounts ./ sum(RAllHistCounts, 2), 'o-', 'linewidth', 1.25, 'markersize', 5);
%     set(gca, 'fontsize', fontSzM, 'fontweight', 'bold', 'ylim', [0 1], 'xlim', RXLims),
%     xlabel(gca, 'activation'), ylabel(gca, 'log density')
%     title(['[', monkey, ssnDate, '] trial-averaged R over subsampled trials'], 'fontweight', 'bold')
%     print('-dtiff', '-r400', [histFigDir, 'J_and_act_avgd_', figNameStem])
%     close
%     
%     %% set up CURBD avg over sets plot
%     % TO DO: CHECK IF nTrlsAvailable == LENGTH(ALLTRLIDS)
%     
%     R_allsets_trunc = cell2mat(R_allsets_trunc_cell'); % (minTrlsAvailablePerSet * nSetsToPlot) x 1 array
%     inds_allsets_trunc = [1; cumsum(repmat(shortestTrl, nTrlsAvailable, 1)) + 1]';
%     [CURBD_allsets, CURBD_allsets_exc, CURBD_allsets_inh, ...
%         avgCURBD_allsets, avgCURBD_allsets_exc, avgCURBD_allsets_inh, ...
%         curbdRgns, nRegionsCURBD] = compute_costa_CURBD(J_allsets, R_allsets_trunc, inds_allsets_trunc, curbdRgns, minTrlsAvailablePerSet * length(setsToPlot));
%     avgCURBD_resh = reshape(avgCURBD_allsets, nRegionsCURBD .^ 2, 1); % for getting YLims
%     avgCURBD_mean = cell2mat(arrayfun(@(i) mean(avgCURBD_resh{i}, 1), 1 : nRegionsCURBD^2, 'un', 0)');
%     CURBDYLims = 0.05; % max(abs(avgCURBD_mean(:))); % max(abs(prctile(avgCURBD_mean(:), [0.5 99.5])));
%     
%     %% TRIAL-AVERAGED MEAN CURRENT CURBD PLOT (ALL SETS)
%     
%     figure('color', 'w');
%     set(gcf, 'units', 'normalized', 'outerposition', [0 0 1 1])
%     count = 1;
%     xTks = round(linspace(1, shortestTrl, 5)); % these will be trial averaged sample points
%     axLPos = linspace(0.03, 0.85, nRegionsCURBD);
%     axBPos = linspace(0.875, 0.06, nRegionsCURBD);
%     axWPos = 0.095;
%     axHPos = 0.095;
%     
%     for iTarget = 1 : nRegionsCURBD
%         nUnitsTarget = numel(curbdRgns{iTarget, 2});
%         
%         for iSource = 1 : nRegionsCURBD
%             
%             subplot(nRegionsCURBD, nRegionsCURBD, count);
%             hold all;
%             count = count + 1;
%             popMeanAvgCURBD = mean(avgCURBD_allsets{iTarget, iSource}, 1);
%             patch([1 : shortestTrl, NaN], [popMeanAvgCURBD, NaN], [popMeanAvgCURBD, NaN], ...
%                 'linewidth', 0.75, 'EdgeColor', 'interp', 'Marker', '.', 'MarkerFaceColor', 'flat')
%             axis tight,
%             line(gca, get(gca, 'xlim'), [0 0], 'linestyle', ':', 'linewidth', 0.75, 'color', [0.2 0.2 0.2])
%             set(gca, 'ylim', [-1 * CURBDYLims, CURBDYLims], 'clim', [-1 * CURBDYLims, CURBDYLims], 'box', 'off', 'tickdir', 'out', 'fontsize', fontSzS)
%             
%             if iTarget == nRegionsCURBD && iSource == 1
%                 xlabel('time (s)', 'fontweight', 'bold');
%                 set(gca, 'xtick', xTks, 'xticklabel', num2str([(xTks * dtData) - dtData]', '%.1f'))
%             else
%                 set(gca, 'xtick', '')
%             end
%             
%             if iSource == 1
%                 ylbl = ylabel([curbdRgns{iTarget, 1}(1 : end - 3), '(', num2str(nUnitsTarget), ')'], 'fontweight', 'bold');
%                 ylbl.Position = [-44.5, 0, -1];
%             end
%             
%             if iSource == 1 && iTarget == nRegionsCURBD
%                 set(gca, 'ytick', [-1 * CURBDYLims, 0, CURBDYLims])
%                 ytickangle(90)
%             else
%                 set(gca, 'ytick', '')
%             end
%             
%             set(gca, 'position', [axLPos(iSource), axBPos(iTarget), axWPos, axHPos])
%             oldOPos = get(gca, 'OuterPosition');
%             set(gca, 'OuterPosition', [1.01 * oldOPos(1), 1.01 * oldOPos(2), 0.99 * oldOPos(3), 0.99 * oldOPos(4)])
%             title([curbdRgns{iSource, 1}(1 : end - 3), ' > ', curbdRgns{iTarget, 1}(1 : end - 3)], 'fontweight', 'bold');
%         end
%     end
%     
%     text(gca, -60, -1.5 * CURBDYLims, [monkey, ssnDate, ' CURBD (trial averaged)'], 'fontsize', fontSzM, 'fontweight', 'bold')
%     cm = brewermap(250,'*RdBu');
%     colormap(cm)
%     print('-dtiff', '-r400', [CURBDFigDir, 'CURBD_avgd_', figNameStem])
%     close
%     
%     %% TRIAL-AVERAGED MEAN CURRENT CURBD PLOT (ALL SETS, EXC)
%     figure('color', 'w');
%     set(gcf, 'units', 'normalized', 'outerposition', [0 0 1 1])
%     count = 1;
%     
%     for iTarget = 1 : nRegionsCURBD
%         nUnitsTarget = numel(curbdRgns{iTarget, 2});
%         
%         for iSource = 1 : nRegionsCURBD
%             
%             subplot(nRegionsCURBD, nRegionsCURBD, count);
%             hold all;
%             count = count + 1;
%             
%             popMeanAvgCURBD_exc = mean(avgCURBD_allsets_exc{iTarget, iSource}, 1);
%             patch([1 : shortestTrl, NaN], [popMeanAvgCURBD_exc, NaN], [popMeanAvgCURBD_exc, NaN], ...
%                 'linewidth', 0.75, 'EdgeColor', 'interp', 'Marker', '.', 'MarkerFaceColor', 'flat')
%             popMeanAvgCURBD_inh = mean(avgCURBD_allsets_inh{iTarget, iSource}, 1);
%             patch([1 : shortestTrl, NaN], [popMeanAvgCURBD_inh, NaN], [popMeanAvgCURBD_inh, NaN], ...
%                 'linewidth', 0.75, 'EdgeColor', 'interp', 'Marker', '.', 'MarkerFaceColor', 'flat')
%             axis tight,
%             line(gca, get(gca, 'xlim'), [0 0], 'linestyle', ':', 'linewidth', 0.75, 'color', [0.2 0.2 0.2])
%             set(gca, 'ylim', [-3 * CURBDYLims, 3 * CURBDYLims], 'clim', [-3 * CURBDYLims, 3 * CURBDYLims], 'box', 'off', 'tickdir', 'out', 'fontsize', fontSzS)
%             
%             if iTarget == nRegionsCURBD && iSource == 1
%                 xlabel('time (s)', 'fontweight', 'bold');
%                 set(gca, 'xtick', xTks, 'xticklabel', num2str([(xTks * dtData) - dtData]', '%.1f'))
%             else
%                 set(gca, 'xtick', '')
%             end
%             
%             if iSource == 1
%                 ylabel([curbdRgns{iTarget, 1}(1 : end - 3), '(', num2str(nUnitsTarget), ')'], 'fontweight', 'bold');
%             end
%             
%             if iSource == 1 && iTarget == nRegionsCURBD
%                 set(gca, 'ytick', [-1 * CURBDYLims, 0, CURBDYLims])
%             else
%                 set(gca, 'ytick', '')
%             end
%             
%             set(gca, 'position', [axLPos(iSource), axBPos(iTarget), axWPos, axHPos])
%             title([curbdRgns{iSource, 1}(1 : end - 3), ' > ', curbdRgns{iTarget, 1}(1 : end - 3)], 'fontweight', 'bold');
%         end
%     end
%     
%     text(gca, -60, -4 * CURBDYLims, [monkey, ssnDate, ' CURBD (trial averaged)'], 'fontsize', fontSzL, 'fontweight', 'bold')
%     cm = brewermap(250,'*RdBu');
%     colormap(cm)
%     print('-dtiff', '-r400', [CURBDFigDir, 'CURBD_avgd_exc_inh_', figNameStem])
%     close
%     
%     %% OVER SETS: set up histogram and CURBD plotting
%     
%     JHistCounts = NaN(length(setsToPlot), nBins);
%     RHistCounts = NaN(length(setsToPlot), nBins);
%     
%     % colorscheme for plotting evolution within set
%     overSsnColors = flipud(winter(length(setsToPlot))); % earlier is green, later blue% flipud(spring(length(setsToPlot))); % earlier is yellow, later is pink
%     
%     avgCURBD_set = cell(1, numel(setsToPlot));
%     
%     count2 = 0;
%     
%     % save J submatrices by set
%     interJOverTrls = NaN(nTrlsAvailable, 1);
%     intraJOverTrls = NaN(nTrlsAvailable, 1);
%     fullJOverTrls = NaN(nTrlsAvailable, 1);
%     for iSet = min(allSetIDs) : setID(nTrlsAvailable)% min(allSetIDs) : max(allSetIDs) % setsToPlot
%         trlIDs = find(setID == iSet);
%         fittedJ_samples = NaN([fittedConsJDim, numel(trlIDs)]);
%         count = 1;
%         % gotta load all J samples
%         for iTrl = 1 : length(trlIDs)
%             mdlfnm = dir([mdlDir, monkey, ssnDate, filesep, 'rnn_', monkey, ssnDate, '_set*_trial', num2str(trlIDs(iTrl)), '.mat']);
%             tmp_mdl = load(mdlfnm.name);
%             mdl = tmp_mdl.RNN.mdl;
%             fittedJ_samples(:, :, :, count) =  mdl.fittedConsJ(:, :, :);
%             count = count + 1;
%             clear mdl
%         end
%         R_set_dims = cell2mat(cellfun(@size, R_U_reordered(trlIDs), 'un', 0));
%         inds_set = [1; cumsum(R_set_dims(:, 2)) + 1]';
%         
%         % reorder full J and R by region (includes intra and inter region)
%         fullJ_samples = fittedJ_samples(newOrder, newOrder, :, :);
%         % get R_set and J_set from loop in previous section
%         J_set = J_U_reordered(:, :, trlIDs);
%         R_set = cell2mat(R_U_reordered(trlIDs)');
%         
%         % get same trials from same sets, but trim down to shortest trial so you can trial average later...
%         R_set_trunc_cell = arrayfun(@(iTrl) R_U_reordered{iTrl}(:, 1 : shortestTrl), trlIDs, 'un', 0);
%         R_set_trunc_mat = cat(3, R_set_trunc_cell{:}); % isequal(R_set_trunc_cell{3}, squeeze(R_set_trunc_mat(:, :, 3)))
%         R_set_trunc = cell2mat(R_set_trunc_cell');
%         inds_set_trunc = [1; cumsum(repmat(shortestTrl, length(trlIDs), 1)) + 1]';
%         
%         % initialize submatrices
%         interJ_samples = fullJ_samples;
%         intraJ_samples = zeros(size(fullJ_samples));
%         
%         % extract submatrices
%         for iRgn = 1 : nRegions
%             in_rgn = newRgnInds(iRgn) + 1 : newRgnInds(iRgn + 1);
%             interJ_samples(in_rgn, in_rgn, :, :) = 0; % set intrargn values to zero
%             intraJ_samples(in_rgn, in_rgn, :, :) = fullJ_samples(in_rgn, in_rgn, :, :); % pull only intrargn values
%         end
%         
%         % grab vals of interest
%         activitySampleAll = NaN(nUnits, nSamples, nTrlsAvailable); % get sample activities
%         for iTrl = 1 : nTrlsAvailable
%             sampleTimePoints = allSPTimePoints(iTrl, :);
%             activitySampleAll(:, :, iTrl) = R_U_reordered{iTrl}(:, sampleTimePoints);
%         end
%         trlIDsCurrSet = allTrialIDs(trlIDs);
%         activitySample = activitySampleAll(:, :, trlIDs);
%         interJOverTrls(trlIDs) = squeeze(mean(interJ_samples, [1 2 3]));
%         intraJOverTrls(trlIDs) = squeeze(mean(intraJ_samples, [1 2 3]));
%         fullJOverTrls(trlIDs) = squeeze(mean(fullJ_samples, [1 2 3]));
%         
%         % want even distribution of sets over fixed dur for each subject/session
%         if ismember(iSet, setsToPlot)
%             count2 = count2 + 1;
%             
%             % CURBD that bitch
%             [CURBD, CURBD_exc, CURBD_inh, avgCURBD, avgCURBD_exc, avgCURBD_inh, ~, ~] = compute_costa_CURBD(J_set, R_set_trunc, inds_set_trunc, curbdRgns, minTrlsAvailablePerSet);
%             avgCURBD_set(count2) = {avgCURBD}; % collect into cell for later plotting of CURBD
%             
%             % get trial averaged truncated activity from minTrlsAvailablePerSet for setsToPlot
%             R_mean = mean(R_set_trunc_mat(:, :, 1 : minTrlsAvailablePerSet), 3);
%             RHistCounts(count2, :) = histcounts(R_mean, RAllHistEdges);
%             
%             % plot of trial-averaged (within this set) fitted Js from the last samplepoint in each trial for
%             % J_source_to_target: reshape(sqrt(nSource) * J, nSource * nTarget, 1)
%             J_mean = mean(J_set(:, :, 1 : minTrlsAvailablePerSet), 3);
%             reshaped_J_mean = reshape(sqrt(nUnits) * J_mean, nUnits .^ 2, 1);
%             [JHistCounts(count2, :)] = histcounts(reshaped_J_mean, JAllHistEdges); % TO DO: look at it with hist and see
%         end
%         
%         %         save([RNNSampleDir, 'classifier_matrices_IntraOnly', monkey, ssnDate, ...
%         %             '_set', num2str(iSet), '.mat'], 'activitySample', 'intraJ_samples', 'trlIDsCurrSet', 'iSet',
%         %             '-v7.3')
%         %
%         %         save([RNNSampleDir, 'classifier_matrices_InterOnly', monkey, ssnDate, ...
%         %             '_set', num2str(iSet), '.mat'], 'activitySample', 'interJ_samples', 'trlIDsCurrSet', 'iSet',
%         %             '-v7.3')
%         %
%         clear interJSample intraJSample activitySample
%     end
%     
%     %% TRIAL-AVERAGED (SPLIT BY SET) J HISTOGRAMS RE-FORMATTING (EACH SET)
%     histFig = figure('color', 'w');
%     set(gcf, 'units', 'normalized', 'outerposition', [0.2 0 0.4 1])
%     
%     if exist('colororder', 'file') ~= 0
%         colororder(overSsnColors);
%     end
%     
%     subplot(2, 1, 1),
%     jHist = semilogy(JAllHistCenters, JHistCounts ./ sum(JHistCounts, 2), 'o-', 'linewidth', 1, 'markersize', 4);
%     set(gca, 'fontsize', fontSzM, 'fontweight', 'bold', 'ylim', [0 1], 'xlim', [-40 40]),
%     xlabel(gca, 'weight'), ylabel(gca, 'log density')
%     arrayfun(@(iLine) set(jHist(iLine), 'MarkerFaceColor', overSsnColors(iLine, :)), 1 : length(jHist));
%     title(['[', monkey, ssnDate, '] trial-averaged J over sets'], 'fontweight', 'bold')
%     legend(gca, cellstr([repmat('set ', length(setsToPlot), 1), num2str(setsToPlot')]), 'location', 'northeastoutside')
%     subplot(2, 1, 2),
%     rHist = semilogy(RAllHistCenters, RHistCounts ./ sum(RHistCounts, 2), 'o-', 'linewidth', 1, 'markersize', 4);
%     set(gca, 'fontsize', fontSzM, 'fontweight', 'bold', 'ylim', [0 1], 'xlim', RXLims),
%     xlabel(gca, 'activation'), ylabel(gca, 'log density')
%     arrayfun(@(iLine) set(rHist(iLine), 'MarkerFaceColor', overSsnColors(iLine, :)), 1 : length(rHist));
%     title(['[', monkey, ssnDate, '] trial-averaged R over sets'], 'fontweight', 'bold')
%     legend(gca, cellstr([repmat('set ', length(setsToPlot), 1), num2str(setsToPlot')]), 'location', 'northeastoutside')
%     
%     print('-dtiff', '-r400', [histFigDir, 'J_and_act_over_sets_', figNameStem])
%     close
%     
%     %% CURBD OVER SETS
%     
%     figure('color', 'w');
%     set(gcf, 'units', 'normalized', 'outerposition', [0 0 1 1])
%     count = 1;
%     xTks = round(linspace(1, shortestTrl, 5)); % these will be trial averaged sample points
%     
%     if exist('colororder', 'file') ~= 0
%         colororder(overSsnColors);
%     end
%     
%     for iTarget = 1 : nRegionsCURBD
%         nUnitsTarget = numel(curbdRgns{iTarget, 2});
%         
%         for iSource = 1 : nRegionsCURBD
%             
%             subplot(nRegionsCURBD, nRegionsCURBD, count);
%             hold all;
%             count = count + 1;
%             
%             popMeanAvgCURBD = mean(avgCURBD_allsets{iTarget, iSource}, 1);
%             popMeanSetCURBD = cell2mat(arrayfun(@(iSet) ...
%                 mean(avgCURBD_set{1, iSet}{iTarget, iSource}, 1), ...
%                 1 : length(setsToPlot), 'un', 0)');
%             plot(1 : shortestTrl, popMeanSetCURBD, '-', 'linewidth', 0.75)
%             axis tight,
%             line(gca, get(gca, 'xlim'), [0 0], 'linestyle', ':', 'linewidth', 0.75, 'color', [0.2 0.2 0.2])
%             set(gca, 'ylim', [-1 * CURBDYLims, CURBDYLims], 'clim', [-1 * CURBDYLims, CURBDYLims], 'box', 'off', 'tickdir', 'out', 'fontsize', fontSzS)
%             
%             if iTarget == nRegionsCURBD && iSource == 1
%                 xlabel('time (s)', 'fontweight', 'bold');
%                 set(gca, 'xtick', xTks, 'xticklabel', num2str([(xTks * dtData) - dtData]', '%.1f'))
%             else
%                 set(gca, 'xtick', '')
%             end
%             
%             if iSource == 1
%                 ylabel([curbdRgns{iTarget, 1}(1 : end - 3), '(', num2str(nUnitsTarget), ')'], 'fontweight', 'bold');
%             end
%             
%             if iSource == 1 && iTarget == nRegionsCURBD
%                 set(gca, 'ytick', [-1 * CURBDYLims, 0, CURBDYLims])
%             else
%                 set(gca, 'ytick', '')
%             end
%             
%             set(gca, 'position', [axLPos(iSource), axBPos(iTarget), axWPos, axHPos]) % same resizing as CURBD_avg plot
%             title([curbdRgns{iSource, 1}(1 : end - 3), ' > ', curbdRgns{iTarget, 1}(1 : end - 3)], 'fontweight', 'bold');
%         end
%     end
%     
%     text(gca, -60, -1.5 * CURBDYLims, [monkey, ssnDate, ' CURBD (over sets)'], 'fontsize', fontSzL, 'fontweight', 'bold')
%     cm = brewermap(100,'*RdBu');
%     colormap(cm)
%     print('-dtiff', '-r400', [CURBDFigDir, 'CURBD_over_sets_', figNameStem])
%     close
%     
%     %% check full, intra, inter
%     
%     JOverTrlsFigDir = [RNNfigdir, 'J_over_trls', filesep]; % pVarOverTrls, first trl convergence, last trl convergence
%     if ~isfolder(JOverTrlsFigDir)
%         mkdir(JOverTrlsFigDir)
%     end
%     
%     trlsPerFig = 500;
%     
%     for f = 1 : ceil(nTrlsAvailable / trlsPerFig) % new plot every five hundred trials
%         figure('color', 'w')
%         set(gcf, 'units', 'normalized', 'outerposition', [0.1 0.25 0.8 0.5])
%         trlsToPlot = (f - 1) * trlsPerFig + 1 : (f * trlsPerFig);
%         
%         if trlsToPlot(end) > nTrlsAvailable
%             trlsToPlot = (f - 1) * trlsPerFig + 1 : nTrlsAvailable;
%         end
%         
%         plot(intraJOverTrls(trlsToPlot), 'b', 'linewidth', 1), hold on,
%         plot(interJOverTrls(trlsToPlot), 'm', 'linewidth', 1)
%         plot(fullJOverTrls(trlsToPlot), 'k', 'linewidth', 1)
%         set(gca, 'fontweight', 'bold', 'fontsize', fontSzS, 'xlim', [1 trlsPerFig])
%         set(gca, 'ylim', [-2.5e-3 2.5e-3], ...
%             'xtick', floor(linspace(1, trlsPerFig, 6)), 'xticklabel', floor(linspace(trlsToPlot(1), trlsToPlot(1) + trlsPerFig, 6)))
%         xlabel('trial#', 'fontsize', fontSzM),
%         legend(gca, {'intra', 'inter', 'full'}, 'location', 'northeast', 'autoupdate', 'off')
%         line(gca, get(gca, 'xlim'), [0 0], 'linestyle', '-.', 'linewidth', 1, 'color', [0.2 0.2 0.2])
%         title(gca, [monkey, ssnDate, ': mean consecutive J over trls'], 'fontsize', fontSzM, 'fontweight', 'bold')
%         
%         print('-dtiff', '-r400', [JOverTrlsFigDir, 'mean_J_over_trls_', figNameStem, '_', num2str(f)])
%         close
%     end
   