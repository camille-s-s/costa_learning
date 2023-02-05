function [RNN, currentParams] = make_RNN_struct(trainRNN, doSmooth, smoothWidth, meanSubtract, doSoftNorm, normByRegion, rmvOutliers, ...
  outliers, rnnDir, RNNname, dtFactor, g, alpha, tauRNN, tauWN, ampInWN, ampWN, nRunTrain, nonlinearity, ...
    train_trl_IDs, test_trl_IDs, nUnits, nTrls, nSets, arrayUnit, arrayRgns, iTrlID, trl_num, setID, rModelSample, tRNN, exp_data, tData, ...
    dtRNN, dtData, w_out, z_out, J, J0, fittedConsJ, sampleTimePoints, cumulative_MSE, mean_MSE, pVars, stdData, ...
    iPredict, ntwk_flavor, n_pred_steps, use_reservoir, use_synthetic_targets, add_readout_unit, add_white_noise, trainAllUnits, nTrlsIncluded, ...
    MAE, MSE, RMSE, readout_targets, readout_predictions) 
% package up and save outputs at the end of training for each link

%% .PARAMS

% doSmooth, 
% smoothWidth,
% meanSubtract,
% doSoftNorm, 
% normByRegion,
% rmvOutliers, 
% outliers,
% rnnDir, 
% RNNname, 
% dtFactor, 
% dtRNN,
% dtData, 
% g, 
% alpha, 
% tauRNN,
% tauWN, 
% ampInWN, 
% nRunTrain, 
% nonlinearity, 
% nUnits, 
% nTrls, 
% nSets, 
% arrayUnit,
% arrayRgns, 

%% .MDL

% iTrlID, 
% trl_num, 
% setID, 
% rModelSample, 
% tRNN % from allPossTS
% exp_data (exp_data_path generates 'outliers', 'fixOnInds', 'stimTimeInds', 'nTrls', 'nTrlsPerSet', 'nSets', 'setID')
% J, 
% J0, 
% fittedConsJ, 
% sampleTimePoints, 
% chi2, 
% pVars, 
% stdData

%%

% add in who called this function for easy identification later
callingFunc = dbstack(1);
called_by = callingFunc.name;

RNN = struct;

currentParams = struct( ...
    'doSmooth', doSmooth, ...
    'smoothWidth', smoothWidth, ...
    'meanSubtract', meanSubtract, ...
    'doSoftNorm', doSoftNorm, ...
    'normByRegion', normByRegion, ...
    'rmvOutliers', rmvOutliers, ...
    'outliers', outliers, ...
    'exp_data_path', [rnnDir 'exp_data' filesep, RNNname, '_exp_data.mat'], ...
    'dtFactor', dtFactor, ...
    'dtRNN', dtRNN, ...
    'dtData', dtData, ...
    'g', g, ...
    'alpha', alpha, ...
    'tauRNN', tauRNN, ...
    'tauWN', tauWN, ...
    'ampInWN', ampInWN, ...
    'ampWN', ampWN, ...
    'nonlinearity', nonlinearity, ...
    'nRunTrain', nRunTrain, ...
    'nUnits', nUnits, ...
    'nTrls', nTrls, ...
    'nSets', nSets, ...
    'arrayUnit', {arrayUnit}, ...
    'arrayRegions', {arrayRgns}, ...
    'ntwk_flavor', ntwk_flavor, ...
    'n_pred_steps', n_pred_steps, ...
    'use_reservoir', use_reservoir, ...
    'use_synthetic_targets', use_synthetic_targets, ...
    'add_readout_unit', add_readout_unit, ...
    'add_white_noise', add_white_noise, ...
    'trainAllUnits', trainAllUnits, ...
    'nTrlsIncluded', nTrlsIncluded, ...
    'train_trl_IDs', train_trl_IDs, ...
    'test_trl_IDs', test_trl_IDs);

if trl_num == 1
    RNN.params = currentParams;
else
    RNN.params = [];
end

if trainRNN
RNN.mdl = struct(...
    'called_by', called_by, ...
    'train_trl', iTrlID, ...
    'trl_num', trl_num, ...
    'setID', setID(iTrlID), ...
    'RMdlSample', rModelSample, ...
    'tRNN', tRNN, ...
    'exp_data', exp_data, ...
    'tData', tData, ...
    'w_out', w_out, ...
    'z_out', z_out, ...
    'J', J, ...
    'J0', J0, ...
    'fittedConsJ', fittedConsJ, ...
    'sampleTimePoints', sampleTimePoints, ...
    'cumulative_MSE_over_runs', cumulative_MSE, ...
    'mean_MSE_over_runs', mean_MSE, ...
    'pVars', pVars, ...
    'stdData', stdData, ...
    'MAE', MAE, ...
    'MSE', MSE, ...
    'RMSE', RMSE, ...
    'iPredict', iPredict, ...
    'readout_targets', readout_targets, ...
    'readout_predictions', readout_predictions);
else
   RNN.test = struct(...
       'called_by', called_by, ...
       'test_trl', iTrlID, ...
       'trl_num', trl_num, ...
       'setID', setID(iTrlID), ...
       'RMdlSample_test', rModelSample, ...
       'tRNN', tRNN, ...
       'exp_data', exp_data, ...
       'tData', tData, ...
       'w_out_test', w_out, ...
       'z_out_test', z_out, ...
       'J_test', J, ...
       'cumulative_MSE_over_steps', cumulative_MSE, ...
       'mean_MSE_over_steps', mean_MSE, ...
       'pVars_test',  pVars, ...
       'stdData_test', stdData, ...
       'MAE_test', MAE, ...
       'MSE_test', MSE, ...
       'RMSE_test', RMSE, ...
       'iPredict', iPredict, ...
       'readout_targets', readout_targets, ...
       'readout_predictions', readout_predictions);
end

end