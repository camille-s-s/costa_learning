% param sweep RNN

% tauRNN tauWN 10*10 possibilities, run with identical parameters
% assess pVar and chi2 as a function of each combination

% param sweep
tauVals = [0.0001, 0.001,0.01];
gVals = [1.1, 1.8, 2.5];

inDir                  = '~/Dropbox/costa_learning/reformatted_data/';
outDir                 = '~/Dropbox/costa_learning/param_sweep/';
cd(inDir)
spikeFiles = dir('spikeData_*.mat');

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

% make wrapper....
for iFile = 1 %:length(spikeFiles)
    fName = spikeFiles(iFile).name;
    fID = fName(strfind(fName, '_') + 1 : end - 4);
    monkey = fID(1);
    ssnDate = fID(2:end);
    
    load(fName)
    
    % get number of units in each array to match to table Hua sent
    arrayList = unique(spikeInfo.array);
    nArray = length(arrayList);
    nInArray = NaN(1, nArray);
    for aa = 1:nArray
        inArray = strcmp(spikeInfo.array,arrayList{aa});
        nInArray(aa) = sum(inArray);
    end
    
    % get match for label to array for this session
    countMatch = arrayfun(@(s) all(ismember(nInArray, rgnCounts(s,:))), 1:size(rgnCounts,1))';
    assert(isequal(rgnCountsTbl(find(countMatch) + 1, 1), {[monkey, ssnDate]}))
    [sortRgnCounts, ix] = sort(rgnCounts(countMatch, :), 'ascend');
    sortRgns = rgns(ix)';
    [sortArrayCounts, ix2] = sort(nInArray, 'ascend');
    sortArrayList = arrayList(ix2);
    
    if ~isequal(sortRgnCounts, sortArrayCounts)
        disp('rgn count mismatch')
        keyboard
    end
    
    arrayRgns = [sortRgns(ix0), sortArrayList(ix0), num2cell(sortRgnCounts(ix0)')];
    
    if params.doSmooth
        mdlName = ['rnn_', fID, ...
            '_smooth_', num2str(params.smoothWidth * params.initBinSize), '_norm_', params.normMethod, '_bin_' num2str(params.binWidth)];
    else
        mdlName = ['rnn_', fID, ...
            '_smooth_', num2str(params.doSmooth), '_norm_', params.normMethod, '_bin_' num2str(params.binWidth)];
    end
    
    pVarVals = NaN(length(tauVals),length(gVals));
    chi2Vals = NaN(length(tauVals),length(gVals));
    runTimes = NaN(length(tauVals),length(gVals));
    
    for iG = 1:length(gVals)
        g = gVals(iG);
        for iTau = 1:length(tauVals)
            tauRNN = tauVals(iTau);
            
            % make fit RNN for first trial
            tic
            RNN = fitCostaRNN(mdlName, ...
                allSpikes, dsSpikeTimes, dsAllEvents, spikeInfo, trlInfo, arrayRgns, ...
                struct( ...
                'tauRNN',tauRNN, ...
                'g', g, ...
                'nRunTrain', 500, ...
                'dtData', params.binWidth, ...
                'normByRegion', false, ...
                'plotStatus', false, ...
                'smoothFlag', false, ...
                'saveMdl', false));
            
            runTimes(iTau, iG) = toc;
            pVarVals(iTau,iG) = RNN.mdl.pVarsTrls(1, end);
            chi2Vals(iTau,iG) = RNN.mdl.chi2Trls(1, end);
            
        end
    end
    
    
    figure,
    imagesc(pVarVals), xlabel('g'), ylabel('tauRNN'), colorbar, colormap(cool), title(['pVar ', monkey, ssnDate])
    set(gca,'fontweight','bold', ...
        'xtick',1:4,'xticklabel', gVals,'ytick',1:4,'yticklabel', tauVals)
    print('-dtiff', '-r400', [outDir, 'pVars_', monkey, ssnDate])
    
    figure,
    imagesc(chi2Vals), xlabel('g'), ylabel('tauRNN'), colorbar, colormap(cool), title(['chi2', monkey, ssnDate])
    set(gca,'fontweight','bold', ...
        'xtick',1:4,'xticklabel', gVals,'ytick',1:4,'yticklabel', tauVals)
    print('-dtiff', '-r400', [outDir, 'chi2_', monkey, ssnDate])
    
end
