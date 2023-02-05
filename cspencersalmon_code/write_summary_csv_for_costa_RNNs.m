% function write_summary_csv_for_costa_RNNs(mdl_paths)
% CS 2023
bd                  = '~/Dropbox (BrAINY Crew)/costa_learning/models/prediction/v20161103';
addpath(genpath(bd)), cd(bd)
mdl_dirs = dir('**/*metrics.mat');
mdl_paths = arrayfun(@(i) [mdl_dirs(i).folder, filesep, mdl_dirs(i).name], 1 : length(mdl_dirs), 'un', 0)';

% exclude any metrics.mat in ARCHIVE subdir
mdl_dirs = mdl_dirs(~cell2mat(arrayfun(@(i) any(strfind(mdl_paths{i}, 'ARCHIVE')), 1 : length(mdl_paths), 'un', 0)'));
mdl_paths = mdl_paths(~cell2mat(arrayfun(@(i) any(strfind(mdl_paths{i}, 'ARCHIVE')), 1 : length(mdl_paths), 'un', 0)'));

% make the header
csv_header_1 = {'parent dir', 'ntwk flavor', '#predicted', '#trials', '#iter', '#pred steps', 'is reservoir', 'use synthetic targets', 'has readout unit', 'noisy update', 'ampInWN', 'smoothwidth', 'tau RNN', 'tau WN', 'dtData', 'dtRNN', 'g', 'P0'};
csv_header_2 = {'MAE (train)', 'MSE (train)', 'RMSE (train)', 'pVar (train)'};
csv_header_3 = {'MAE (test)', 'MSE (test)', 'RMSE (test)', 'pVar (test)'};
csv_header = [csv_header_1, csv_header_2, csv_header_3];

% set up
n_rows = length(mdl_paths);
n_cols = length(csv_header);
csv_vals = cell(n_rows, n_cols);

% load all versions (each is a row)
for iMdl = 1 : length(mdl_paths) 
    metrics_file = mdl_paths{iMdl};
    parent_dir = mdl_dirs(iMdl).folder(max(strfind(mdl_dirs(iMdl).folder, filesep)) + 3 : end);
    
    load(metrics_file)
    P = currentParams;
    
    % TO DO: ADD TRIAL RANGE!!!!
    all_trls = [P.train_trl_IDs, P.test_trl_IDs];
    trl_range = [min(all_trls) max(all_trls)];
    
    % older structs may not have these fields
    if ~isfield(P, 'add_readout_unit') % default is no
        has_readout_unit = false;
    else
        has_readout_unit = P.add_readout_unit;
    end
    if ~isfield(P, 'add_white_noise') % default is no
        noisy_update = false;
    else
        noisy_update = P.add_white_noise;
    end

    % grab values for this row
    csv_vals_1 = {parent_dir, P.ntwk_flavor,  P.nUnits, P.nTrlsIncluded, P.nRunTrain, P.n_pred_steps, P.use_reservoir, P.use_synthetic_targets, has_readout_unit, noisy_update, P.ampInWN, P.smoothWidth, P.tauRNN, P.tauWN, P.dtData, P.dtRNN, P.g, P.alpha};
    csv_vals_2 = {MAE_train, MSE_train, RMSE_train, pVar_train};
    csv_vals_3 = {MAE_test, MSE_test, RMSE_test, pVar_test};
    curr_row_vals = [csv_vals_1, csv_vals_2, csv_vals_3];
    csv_vals(iMdl, :) = curr_row_vals;
end

CSV = [csv_header; csv_vals];
writecell(CSV, [pwd filesep 'v20161103_comparative_metrics', '.csv'])

% end