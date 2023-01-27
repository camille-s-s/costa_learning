function [train_trl_IDs, test_trl_IDs, nTrlsTrain, nTrlsTest, start_trl_num, prevJ, prev_w_out, trainRNN, iPredict] = get_train_test_lists_and_progress(rnnDir, rnnSubDir, RNNname, nTrlsIncluded, nUnits, trainAllUnits, nUnitsIncluded)
% input: rnnDir, RNNname, nTrlsIncluded, trainAllUnits
% output: train_trl_IDs, test_trl_IDs, start_trl_num, prevJ, trainRNN

% define train/test split
nTrlsTrain = round(0.75 * nTrlsIncluded);
nTrlsTest = nTrlsIncluded - nTrlsTrain;

% different naming structure if fitting RNNs for prediction or regular version
callingFunc = dbstack(1);
called_by = callingFunc.name;

 % initialize list if starting at beginning of a file
if ~isfile([rnnDir, 'train_test_lists' filesep, RNNname, '_train_test_list.mat']) 
    subset_trl_IDs = randperm(nTrlsIncluded, nTrlsIncluded); % 2022-10-26 temporary, for dev. will be randperm(nTrlsIncluded); or randperm(nTrls, nTrlsIncluded)
    train_trl_IDs = subset_trl_IDs(1 : nTrlsTrain); 
    test_trl_IDs = subset_trl_IDs(nTrlsTrain + 1 : end);
    
    if trainAllUnits
        iPredict = 1 : nUnits;
    else
        iPredict = sort(randperm(nUnits, nUnitsIncluded)); % nUnitsIncluded unique indices selected randomly from 1 : nUnits
    end
    
    save([rnnDir, 'train_test_lists' filesep, RNNname, '_train_test_list.mat'], 'train_trl_IDs', 'test_trl_IDs', 'iPredict')
    start_trl_num = 1;
    trainRNN = true;

% if list already initialized, check to see how far you are in it, and load iPredict as well
else
    load([rnnDir, 'train_test_lists' filesep, RNNname, '_train_test_list.mat'], 'train_trl_IDs', 'test_trl_IDs', 'iPredict')
    if strcmp(called_by, 'fit_costa_RNN_prediction') || strcmp(called_by, 'fit_costa_RNN_1step_prediction') % if prediction
        prevMdls = dir([rnnSubDir, RNNname, '_train*_num*.mat']);
        trl_IDs_in_dir = arrayfun(@(i) ...
            str2double(prevMdls(i).name(strfind(prevMdls(i).name, 'train') + 5 : strfind(prevMdls(i).name, '_num') - 1)), ...
            1 : length(prevMdls));
    elseif strcmp(called_by, 'fit_costa_RNN_v3') % if descriptive
        prevMdls = dir([rnnSubDir, RNNname, '_set*_trl*.mat']);
        trl_IDs_in_dir = arrayfun(@(i) ...
            str2double(prevMdls(i).name(strfind(prevMdls(i).name, 'trl') + 3 : strfind(prevMdls(i).name, '.') - 1)), ...
            1 : length(prevMdls));
    else
        keyboard;
    end
    
    % see which of the trial IDs in the train list are already in the directory
    last_completed_trl_num = find(~ismember(train_trl_IDs, trl_IDs_in_dir), 1, 'first') - 1;
    start_trl_num = last_completed_trl_num + 1;
    
    % if done training, move on to testing
    if isempty(last_completed_trl_num) % start_trl_num will also be empty
        last_completed_trl_num = nTrlsTrain;
        start_trl_num = last_completed_trl_num + 1;
        trainRNN = false;
    else
        trainRNN = true;
    end
end

% if partially trained before, load last J
if start_trl_num > 1
    
    if strcmp(called_by, 'fit_costa_RNN_prediction') || strcmp(called_by, 'fit_costa_RNN_1step_prediction') % if predictive
        prevMdl = load([rnnSubDir, dir([rnnSubDir, RNNname, ...
            '_train', num2str(train_trl_IDs(last_completed_trl_num)), '_num', num2str(last_completed_trl_num), '.mat']).name]);
    elseif strcmp(called_by, 'fit_costa_RNN_v3') % if descriptive
          prevMdl = load([rnnSubDir, dir([rnnSubDir, RNNname, ...
        '_set*_trl', num2str(train_trl_IDs(last_completed_trl_num)), '.mat']).name]);
    else
        keyboard;
    end
    
    prevJ = prevMdl.RNN.mdl.J;
    try
        prev_w_out = prevMdl.RNN.mdl.w_out;
    catch
        prev_w_out = []; % backwards compatibility
    end
    % iPredict = prevMdl.RNN.mdl.iPredict;
    prev_trl_ID = prevMdl.RNN.mdl.train_trl;
    assert(prev_trl_ID == train_trl_IDs(last_completed_trl_num))
    
else
    
    prevJ = [];
    prev_w_out = [];
    % iPredict = [];
    
end

end