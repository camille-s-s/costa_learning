function [train_trl_IDs, test_trl_IDs, nTrlsTrain, nTrlsTest, start_trl_num, prevJ, trainRNN] = get_train_test_lists_and_progress(rnnDir, rnnSubDir, RNNname, nTrls, nTrlsIncluded)
% input: rnnDir, RNNname, nTrls, nTrlsIncluded
% output: train_trl_IDs, test_trl_IDs, start_trl_num, prevJ, trainRNN

% define train/test split
nTrlsTrain = round(0.75 * nTrlsIncluded);
nTrlsTest = nTrlsIncluded - nTrlsTrain;

if ~isfile([rnnDir, 'train_test_lists' filesep, RNNname, '_train_test_list.mat']) % initialize list if starting at beginning
    subset_trl_IDs = randperm(nTrls, nTrlsIncluded);
    train_trl_IDs = subset_trl_IDs(1 : nTrlsTrain); % isequal(all_trl_IDs(ismember(all_trl_IDs, train_trl_IDs)), train_trl_IDs)
    test_trl_IDs = subset_trl_IDs(nTrlsTrain + 1 : end); % all_trl_IDs(~ismember(all_trl_IDs, sort(train_trl_IDs)));
    save([rnnDir, 'train_test_lists' filesep, RNNname, '_train_test_list.mat'], 'train_trl_IDs', 'test_trl_IDs')
    start_trl_num = 1;
else
    load([rnnDir, 'train_test_lists' filesep, RNNname, '_train_test_list.mat'], 'train_trl_IDs', 'test_trl_IDs')
    prevMdls = dir([rnnSubDir, RNNname, '_train_trl*_num*.mat']);
    trl_IDs_in_dir = arrayfun(@(i) ...
        str2double(prevMdls(i).name(strfind(prevMdls(i).name, 'trl') + 3 : strfind(prevMdls(i).name, '_num') - 1)), ...
        1 : length(prevMdls));
    
    % see which of the trial IDs in the train list are already in the directory
    last_completed_trl_num = find(~ismember(train_trl_IDs, trl_IDs_in_dir), 1, 'first') - 1;
    start_trl_num = last_completed_trl_num + 1;
    
    % if done training, move on to testing
    if isempty(last_completed_trl_num) % start_trl_num will also be empty
        last_completed_trl_num = nTrlsTrain;
        trainRNN = false;
    else
        trainRNN = true;
    end 
end

% if partially trained before, load last J
if start_trl_num > 1
    prevMdl = load([rnnSubDir, dir([rnnSubDir, RNNname, ...
        '_train_trl', num2str(train_trl_IDs(last_completed_trl_num)), '_num', num2str(last_completed_trl_num) '.mat']).name]);
    prevJ = prevMdl.RNN.mdl.J;
    prev_trl_ID = prevMdl.RNN.mdl.train_trl;
    assert(prev_trl_ID == train_trl_IDs(last_completed_trl_num))
else
    prevJ = [];
end

end