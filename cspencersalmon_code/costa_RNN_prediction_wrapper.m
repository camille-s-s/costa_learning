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

% set_num_cores(4);

% get min trial length across sessions for consecutive fitted J sampling by trial
if ~isfile([inDir 'minLenFromStimAllFiles.mat'])
    minLenFromStim = NaN(length(spikeFiles), 1);
    
    for iFile = 1 : length(spikeFiles)
        fName = spikeFiles(iFile).name;
        fID = fName(1:strfind(fName, '_') - 1);
        monkey = fID(1);
        ssnDate = fID(2:end);
        S = load([monkey, ssnDate, '_spikesCont']);
        M = load(fName);
        allSpikes = S.dsAllSpikes;
        allEvents = M.dsAllEvents;
        fixOnInds = [find(allEvents == 1), size(allSpikes, 2)];
        stimTimeInds = find(allEvents == 2);
        minLenFromStim(iFile) = min([fixOnInds(2:end)' - stimTimeInds']); % minimum time between next fixation and current stim time
        clearvars S M
    end
    
    minLenFromStimAll = min(minLenFromStim);
    save([inDir 'minLenFromStimAllFiles.mat'], 'minLenFromStimAll')
else
    load([inDir 'minLenFromStimAllFiles.mat'], 'minLenFromStimAll')
end

% wrapper
for iFile = 1 % : 3 % : length(spikeFiles) % 3 4] %:length(spikeFiles)
    fName = spikeFiles(iFile).name;
    fID = fName(1:strfind(fName, '_') - 1);
    monkey = fID(1);
    ssnDate = fID(2:end);
    S = load([monkey, ssnDate, '_spikesCont']);
    M = load(fName);
    
    % grab only necessary metainfo
    arrayUnit = M.spikeInfo.array;
    allPossTS = M.dsSpikeTimes;
    allEvents = M.dsAllEvents;
    trlInfo = M.trlInfo;
    
    % grab only necessary spike info
    allSpikes = S.dsAllSpikes;
    params = cell2struct(S.params(:,2), S.params(:,1));
    dtData = params.binWidth;

    % get number of units in each array and match labels to regions by monkey
    switch monkey
        case 'v' % voltaire
            arrayList = {'B', 'F', 'A', 'E', ...
                'G', 'C', 'D', 'H'};
        case 'w' % waldo
            arrayList = {'A', 'F', 'B', 'E', ...
                'H', 'G', 'C', 'D'};
        otherwise
            disp('unknown monkey/filename')
            keyboard
    end
    
    nArray = length(arrayList);
    nInArray = arrayfun(@(a) sum(strcmp(arrayUnit, arrayList{a})), 1 : nArray); 
    inArrays = arrayfun(@(aa) strcmp(arrayUnit, arrayList{aa}), 1:nArray, 'un', false)';

    % each row is region label, region letter, and logical
    arrayRgns = [rgns', arrayList', inArrays];
    
    mdlName = ['rnn_', fID];
    
    fit_costa_RNN_prediction(mdlName, ...
        allSpikes, allPossTS, allEvents, arrayUnit, trlInfo, arrayRgns, ...
        struct( ...
        'g', 1.5, ...
        'tauRNN', 0.001, ...
        'dtFactor', 20, ...
        'trainFromPrev', true, ...
        'trainAllUnits', false, ...
        'nRunTrain', 500, ... 
        'tauWN', NaN, ...
        'ampInWN', NaN, ...
        'alpha', 1, ...
        'dtData', dtData, ...
        'minLen', minLenFromStimAll, ...
        'doSmooth', true, ...
        'smoothWidth', 0.20, ...
        'rmvOutliers', true, ...
        'meanSubtract', false, ...
        'doSoftNorm', true, ...
        'plotStatus', true, ...
        'mouseVer', ['', 'prediction'], ...
        'saveMdl', true));
end