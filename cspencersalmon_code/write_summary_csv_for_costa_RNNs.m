function write_summary_csv_for_costa_RNNs(mdl_paths)
% CS 2023
% bd                  = '~/Dropbox (BrAINY Crew)/costa_learning/models/prediction/v20161103';
% addpath(genpath(bd))
% cd(bd)
% mdl_dirs = dir('**/*metrics.mat');
% mdl_paths = arrayfun(@(i) [mdl_dirs(i).folder, filesep, mdl_dirs(i).name], 1 : length(mdl_dirs), 'un', 0)';
%
% make the header
csv_header_1 = {'ntwk flavor', '#predicted', '#pred steps', 'reservoir', 'synthetic targets', 'ampInWN', '#trials', 'smooth width', 'tau RNN', 'tau WN', '#iterations', 'dtData', 'dtRNN', 'g', 'P0'};
csv_header_2 = {'MAE (train)', 'MSE (train)', 'RMSE (train)', 'pVar (train)', 'avg pVar (train)', 'avg MSE (train)', 'avg sum MSE (train)'};
csv_header_3 = {'MAE (test)', 'MSE (test)', 'RMSE (test)', 'pVar (test)', 'avg pVar (test)', 'avg MSE (test)', 'avg sum MSE (test)'};
csv_header = [csv_header_1, csv_header_2, csv_header_3];
% set up
n_rows = length(mdl_paths);
n_cols = length(csv_header);
csv_vals = cell(n_rows, n_cols);
% load all versions (each is a row)
for iMdl = 1 : length(mdl_paths) 
    metrics_file = mdl_paths{iMdl};
    load(metrics_file)
    P = currentParams;
    % grab values for this row
    csv_vals_1 = {P.ntwk_flavor,  P.nUnits, P.n_pred_steps, P.use_reservoir, P.use_synthetic_targets, P.ampInWN, P.nTrlsIncluded, P.smoothWidth, P.tauRNN, P.tauWN, P.nRunTrain, P.dtData, P.dtRNN, P.g, P.alpha};
    csv_vals_2 = {MAE_train, MSE_train, RMSE_train, pVar_train, avg_final_pVar_train, avg_final_mean_MSE_train, avg_final_sum_MSE_train};
    csv_vals_3 = {MAE_test, MSE_test, RMSE_test, pVar_test, avg_final_pVar_test, avg_final_mean_MSE_test, avg_final_sum_MSE_test};
    curr_row_vals = [csv_vals_1, csv_vals_2, csv_vals_3];
    csv_vals(iMdl, :) = curr_row_vals;
end

CSV = [csv_header; csv_vals];
writecell(CSV, [pwd filesep 'v20161103_comparative_metrics', '.csv'])

end