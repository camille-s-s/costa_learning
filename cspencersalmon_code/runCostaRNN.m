bd                  = '~/Dropbox (BrAINY Crew)/costa_learning/';
inDir               = [bd 'reformatted_data/'];

addpath(genpath(bd))
cd(inDir)
spikeFiles = dir('*_meta.mat');

bad_files = arrayfun(@(iFile) any(strfind(spikeFiles(iFile).name, '._')), 1:length(spikeFiles));
spikeFiles(bad_files) = [];

% set up table for matching arrays to regions
rgns = {'left_rdLPFC', 'left_mdLPFC', 'left_cdLPFC', 'left_vLPFC', ...
    'right_rdLPFC', 'right_mdLPFC', 'right_cdLPFC', 'right_vLPFC'};
rgnCountsTblHdr = ['session', rgns];
rgnCountsTbl = [rgnCountsTblHdr; ...
    {'w20160205', 74, 49, 134, 2, 72, 97, 96, 63; ...
    'w20160210', 80, 92, 180, 7, 60, 125, 127, 58; ...
    'w20160211', 66, 79, 143, 3, 75, 126, 131, 59; ...
    'v20161103', 66, 48, 57, 64, 1, 65, 82, 72; ...
    'v20161104', 63, 44, 70, 63, 1, 81, 102, 80; ...
    'v20161107', 69, 78, 130, 58, 8, 96, 123, 136}];
rgnCounts = cell2mat(rgnCountsTbl(2:end,2:end));
[~, ix0] = sort(rgns);

set_num_cores(length(spikeFiles) * 2);

% make wrapper....
for iFile = 1:length(spikeFiles)
    fName = spikeFiles(iFile).name;
    fID = fName(1:strfind(fName, '_') - 1);
    monkey = fID(1);
    ssnDate = fID(2:end);
    S = load([monkey, ssnDate, '_spikesCont']);
    M = load(fName);
    
    % reformat params for fitRNN
    params = cell2struct(S.params(:,2), S.params(:,1));
    
    % get number of units in each array to match to table Hua sent
    arrayList = unique(M.spikeInfo.array);
    nArray = length(arrayList);
    nInArray = NaN(1, nArray);
    for aa = 1:nArray
        inArray = strcmp(M.spikeInfo.array,arrayList{aa});
        nInArray(aa) = sum(inArray);
    end
    
    % get match for label to array for this session
    countMatch = arrayfun(@(s) all(ismember(nInArray, rgnCounts(s,:))), 1:size(rgnCounts,1))';
    assert(isequal(rgnCountsTbl(find(countMatch) + 1, 1), {[monkey, ssnDate]}))
    [sortRgnCounts, ix] = sort(rgnCounts(countMatch, :), 'ascend');
    sortRgns = rgns(ix)';
    [sortArrayCounts, ix2] = sort(nInArray, 'ascend');
    sortArrayList = arrayList(ix2);
    
    inSortedArrays = arrayfun(@(aa) strcmp(M.spikeInfo.array,sortArrayList{aa}), 1:nArray, 'un', false)';
    
    if ~isequal(sortRgnCounts, sortArrayCounts)
        disp('rgn count mismatch')
        keyboard
    end
    
    % arrayRgns = [sortRgns(ix0), sortArrayList(ix0), num2cell(sortRgnCounts(ix0)')];
    arrayRgns = [sortRgns(ix0), sortArrayList(ix0), inSortedArrays(ix0)];
    
    mdlName = ['rnn_', fID];
    
    % make fit RNN
    disp(['    FITTING: ' monkey, ssnDate, ''])
    tic
    
    RNN = fitCostaRNN(mdlName, ...
        S.dsAllSpikes, M.dsSpikeTimes, M.dsAllEvents, M.spikeInfo, M.trlInfo, arrayRgns, ...
        struct( ...
        'trainFromPrev', true, ...
        'dtData', params.binWidth, ...
        'plotStatus', false, ...
        'saveMdl', true));
    toc
    
    clear S
end