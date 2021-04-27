function extractSpikes(params)
% DESCRIPTION
%
% Loop through all Costa neurophys data and output spikeData (variables)
% and tables spikeInfo and trlInfo for indexing into them for future
% analyses. See cs_ref for more details. Additionally, plot heatmaps and
% population trial-averaged mean activity for visualization.
%
% OUTPUTS
%
% spikeData: Contains the following variables for a given session, monkey,
% and NIP:
%   alignedBinnedSpikes: my version, #neurons x #trials #bins
%   params: params for the above two variables
%   spikeCount: #neurons x #trials #bins, from #YYYYMMDD_nip#.mat
%   x spikesSDF: #neurons x #trials #bins, from #YYYYMMDD_nip#.mat (all
%   NaNs, not included for now)
%   x alignedSpikes: my version, #neurons x #trials x #samplepoints (too
%   big, no longer included)
%
% spikeInfo: table (it's just the Neurons variable from the
% #YYYYMMDD_nip#.mat files, but labeled more clearly) for a given session,
% monkey, and NIP
%
% trlInfo: table from Coder and Coder.BHVOut variables, labeled more
% clearly, for a given session, monkey, and NIP. See trlInfoVarNames for
% more detailed description.

%% TO DO

% 1. confirm about msec assumption on timings and spikes confirm about
% sampling. CHECK!!! paper says 1 kHz (1000/s so 0.001 place but this seems
% to be in 10 kHz. Assuming that this is in fractions of a second, based on
% number of decimal places in aTS and event times.
% 2. do we bin then smooth or smooth then bin and if so how do we smooth?
% currently within neuron: smoothing and binning. (then trial average then
% normalize??)
% 3. should I normalize after trial averaging? and if so how?
% 4. TO DO: IDK HOW TO PICK kernel for smoothing
% 5. % TO DO: what are all the regions and how to index uniquely by
% conditions?
% 6. % TO DO: in end, gonna trial average within trial types.....I guess for
% each session? it's not the same units for every session, is it? how to
% get unique trial types?
% 7. Why is spikesSDF all NaNs
% 8. Confirm: is Coder.orientation referring to chosen stim?

%% housekeeping

clearvars;
close all

%% input params

% data specs
monkeyList              = {'v', 'w'};
nipList                 = [1, 2];
dateFormat              = 'yyyymmdd';

% data processing params
normMethod              = 'none';           % zscore, maxscale, none
doSmooth                = true;             % true or false
smoothWinSteps          = 20;               % in sample points, can be any number < dimensions of extracted window
lockEvent               = 'fixation';       % fixation, stim, choice, reward
winBounds               = [0 3];            % in sec around lockEvent
extractWholeTrial       = false;             % overwrite winBounds if need be
initBinSize             = 0.001;            % in msec; a function of original data sampling rate

% derived from data processing params
binSizeStr = num2str(initBinSize);
numDP = numel(binSizeStr(strfind(binSizeStr,'.')+1:end));

% overwrite defaults based on inputs
if exist('params','var')
    assignParams(who,params);
end

% package up params for reference
params.monkeyList       = monkeyList;
params.nipList          = nipList;
params.normMethod       = normMethod;
params.doSmooth         = doSmooth;
params.smoothWinSteps   = smoothWinSteps;
params.lockEvent        = lockEvent;
params.winBounds        = winBounds;
params.initBinSize      = initBinSize;
params.numDP            = numDP;

paramStr = ['_smth_', num2str(doSmooth)', '_win_', num2str(smoothWinSteps), '_norm_' normMethod, '_bnstp_' num2str(smoothWinSteps)];

%% set up

coderDir = '~/Dropbox/costa_learning/data/coder_files/';
nphysDir = '~/Dropbox/costa_learning/data/neurophys_data/';
figDir = '~/Dropbox/costa_learning/figures/';
outDir = '~/Dropbox/costa_learning/reformatted_data/';

cd(coderDir)
coderFiles = dir('*1.mat');
cd(nphysDir)
nphysFiles = dir('*1.mat');

% name parsing depends on length of these names all being consistent
coderNameLength = arrayfun(@(x) length(coderFiles(x).name), 1:length(coderFiles));
assert(numel(unique(coderNameLength))==1)
nphysNameLength = arrayfun(@(x) length(nphysFiles(x).name), 1:length(nphysFiles));
assert(numel(unique(nphysNameLength))==1)

trlInfoVarNames = {'trls_since_nov_stim', ... % trlssincenov: #trials since novel stimulus, ...
    'nov_stim_idx', ... % picid: (when & where the novel image appeared)
    'nov_stim_loc', ... % dropin_loc (novel trial' location )
    'nov_stim_rwd_prob', ... % dropinrewprob (novel trial' reward probability )
    'stimID', ... % stim: stimID (it's an actual value but TO DO: find out what the actual value means)
    'reward_probs', ... % reward probability
    'choice_idx', ... % choices: which stim was chosen, will be 1, 2, or 3 (it's an index)
    'chosen_stim_orientation', ... % orientation: orientation of the (chosen) stimulus triangle (up or down) TO DO: IS THIS FOR THE CHOSEN STIM OR WHAT??? WHAT'S THIS TRIANGLE???
    'chosen_stim_dir', ... % direction: direction of the chosen stimulus (1 - 6)
    'chosen_stim', ... % chosenstim: Coder.stim(Coder.choices) = Coder.chosenstim (or, stimID(choice_idx) = chosen_stim)
    'chose_nov_stim', ... % chosenovel (whether chose the novel image)
    'reward', ... % reward: got a reward or not
    'event_times', ... % fixlock, stimlock, choice time, juicelock
    'aligned_event_times'}; % the above but with lockEventTime subtracted

spikeInfoVarNames = {'monkey', 'array', 'date', 'nip', 'electrode', 'unit', 'port'};

%% Loop through neurophys and behavioral files to extract all spike info we
% might care about and put all event info we might care about into a
% helpful table. For my own edification extract spikes as well

alignedBinnedSpikes = [];
spikeInfo = [];
trlInfo = [];
for iCoder = 1:length(coderFiles)
    coderfName = coderFiles(iCoder).name;
    brks = strfind(coderfName, '_');
    
    monkey = coderfName(brks(1)+1);
    sessionDate = coderfName(brks(2)+1:brks(3)-1);
    
    for iNip = [1 2]
        nipNum = iNip; % nip2 is same session same day same monkey just different machine with
        % a SLIGHT difference in clock (~10ms) so we gotta pull its spikes
        % separately and then stick em together
        % coderNameParsed = ['novelty_', monkey, '_', sessionDate, '_coder_nip' num2str(nipNum) '.mat'];
        % assert(isequal(coderNameParsed,coderfName)) % double check same format
        
        nphysNameParsed = [monkey, sessionDate, '_nip', num2str(nipNum), '.mat'];
        load([nphysDir, nphysNameParsed], 'aTS', 'Bin', 'Coder', 'Neurons', 'spikeCount')
        
        switch lockEvent
            case 'fixation'
                lockEventTime = Coder.fixlock;
            case 'stim'
                lockEventTime = Coder.stimlock;
            case 'choice'
                lockEventTime = Coder.stimlock + Coder.srt./1000;
            case 'reward'
                lockEventTime = Coder.juicelock; % if aligned on choice, this will also be aligned
        end
        
        assert(isequal(Coder.validtrls, find(Coder.BHVerror==0))) % make sure we are extracting non error trials only
        
        % extract trial info
        eventTimes = [Coder.fixlock, Coder.stimlock, Coder.stimlock + Coder.srt./1000, Coder.juicelock];
        eventTimesAligned = eventTimes - lockEventTime;
        
        assert(all(all(diff(eventTimes, [],2)>=0))) % make sure these are happening in the correct order
        
        eventTimes = round(eventTimes, numDP, 'decimals');
        eventTimesAligned = round(eventTimesAligned, numDP, 'decimals');
        

        % make sure trials are not overlapping (end of previous trial must be before start of current)
        minDurBtwnTrls = min(Coder.fixlock(2:end) - (Coder.fixlock(1:end-1)));
        
        if extractWholeTrial % from lock event start
            winBounds(1) = 0; % overwrite if necessary
            winBounds(2) = round(0.995*minDurBtwnTrls,2,'decimals');
        else
            assert(range([0 winBounds(2)]) <= minDurBtwnTrls)
        end
        
        % prep for binning
        nTrls = length(Coder.validtrls);
        nUnits = size(Neurons,1);
        coderBinSteps = Bin.window/initBinSize; % in steps
        nBins = round((range(winBounds) + 3 * Bin.window) ./ Bin.window);
        
        % outputs for this session
        rrprob_aval = Coder.BHVout(:,15:17);
        nov_stim_loc = Coder.BHVout(:,26);
        nov_stim_rwd_prob = Coder.BHVout(:,27);
        chose_nov_stim = Coder.BHVout(:,31);
        picID = arrayfun(@(iTrl) find(Coder.picid(iTrl,:)), 1:nTrls, 'UniformOutput', 0)';
        picID(cellfun(@isempty,picID))={0};
        picID = cell2mat(picID);
        
        % TO DO: MAKE SURE ALL OF THIS EXCEPT EVENT TIMES ARE THE SAME
        % BETWEEN NIPS
        trlInfo_curr = table(Coder.trlssincenov, ...
            picID, ...
            nov_stim_loc, ...
            nov_stim_rwd_prob, ...
            Coder.stim, ...
            rrprob_aval, ...
            Coder.choices, ...
            Coder.orientation, ...
            Coder.direction, ...
            Coder.chosenstim, ...
            chose_nov_stim, ...
            Coder.reward, ...
            eventTimes, ...
            eventTimesAligned, ...
            'VariableNames', trlInfoVarNames);
        
        spikeInfo_curr = cell2table(Neurons, 'VariableNames', spikeInfoVarNames);
        
        % how this is rounded changes how precise our bin size can be - not
        % sure if this is artifact or real
        lockEventTime = round(lockEventTime, numDP, 'decimals');
        
        % bounds for collecting spikes - with a little extra so winBounds are
        % actually win centers, as in Bin.cen
        iStart = winBounds(1) - (1.5 * Bin.window); % + initBinSize;
        iStop = winBounds(2) + (1.5 * Bin.window);
        
        % for plotting
        edges = (iStart:(iStop - iStart)/nBins:iStop)';
        cen = edges(1:end-1) + (diff(edges) ./ 2);
        nBins = nBins - 2;
        
        % align ith unit's spikes around lockEvent for all trials
        winTimes = [lockEventTime+iStart, lockEventTime+iStop]; % bounds for all trials for ith unit
        
        alignedBinnedSpikes_curr = NaN(nUnits, nTrls, nBins);
        % alignedSpikes = NaN(nUnits, nTrls, nBins*coderBinSteps);
        
        tic
        for nn = 1:nUnits % For each unit in file
            aTSUnit = aTS{nn};
            
            allSpikeTS = round(aTSUnit, numDP, 'decimals'); % all of a given unit's spike timestamps
            
            % timestamps for spikes within bounds
            winSpikeTS = arrayfun(@(iTrl) allSpikeTS ...
                (allSpikeTS > winTimes(iTrl,1) ...
                & allSpikeTS <= winTimes(iTrl,2)), ...
                1:nTrls, 'UniformOutput', false)';
            
            % count all spikes within winTimes for each trial for ith unit
            N = cell2mat(arrayfun(@(iTrl) ...
                histcounts(winSpikeTS{iTrl}, winTimes(iTrl,1):initBinSize:winTimes(iTrl,2)), ...
                1:nTrls, 'UniformOutput', false)'); % trials x #made-up sample points for ith unit
            
            % smooth by kernel as wide as their binning window
            if doSmooth
                N = smoothdata(N, 2, 'movmean', smoothWinSteps);
            end
            
            % trim off the extra bins at start and end
            N = N(:, coderBinSteps:end - coderBinSteps - 1);
            
            % normalize
            switch normMethod
                case 'zscore'
                    N = zscore(N, 0, 2);
                case 'maxscale'
            end
            
            % bin aligned spikes (#trials x #bins for ith unit)
            %if doSmooth % if smoothing it's a mean over a window
            %binnedN = cell2mat(arrayfun(@(i) mean(N(:,i:i+coderBinSteps-1),2), 1:coderBinSteps:size(N,2)-coderBinSteps+1, 'UniformOutput', 0));
            %else % if no smooth it's a count
            binnedN = cell2mat(arrayfun(@(i) sum(N(:,i:i+coderBinSteps-1),2), 1:coderBinSteps:size(N,2)-coderBinSteps+1, 'UniformOutput', 0));
            %end
            
            alignedBinnedSpikes_curr(nn, :, :) = binnedN; % binnedN(:, cen>=winBounds(1) & cen<=winBounds(2));
            % alignedSpikes(nn, :, :) = N;
        end
        
        toc
        
        cen = cen(cen>=winBounds(1) & cen<=winBounds(2));
        
        % concatenate nips along #neurons dimension (#trials and #bins
        % should be the same)
        alignedBinnedSpikes = cat(1,alignedBinnedSpikes,alignedBinnedSpikes_curr);
        spikeInfo = [spikeInfo; spikeInfo_curr];
        trlInfo{iNip} = trlInfo_curr;
        
        
    end
    
    
    %% extract variables for plotting by array
    
    arrayList = unique(Neurons(:,2));
    nArray = length(arrayList);
    
    % extract metrics for each array
    popTrlMean = cell(2, nArray);
    trlMean = cell(2, nArray);
    nUnitsArray = NaN(1,nArray);
    for aa = 1:nArray
        inArray = strcmp(spikeInfo.array,arrayList{aa});
        nUnitsArray(aa) = sum(inArray);
        arraySpikes = alignedBinnedSpikes(inArray,:,:);
        arraySpikeCount = spikeCount(inArray,:,:);
        % arraySDF = spikesSDF(inArray,:,:);
        
        % since these variables are in mean spikes / bin, convert to Hz by
        % multiplying by (1 / Bin.window) bins / sec
        popTrlMean{1, aa} = squeeze(mean(arraySpikes, [1 2])) .* (1 / Bin.window);
        popTrlMean{2, aa} = squeeze(mean(arraySpikeCount, [1 2])) .* (1 / Bin.window);
        % popTrlMean{3, aa} = squeeze(mean(arraySDF, [1 2])) .* (1 /
        % Bin.window)';
        
        trlMean{1, aa} = squeeze(mean(arraySpikes, 2)) .* (1 / Bin.window);
        trlMean{2, aa} = squeeze(mean(arraySpikeCount, 2)) .* (1 / Bin.window);
        
    end
    
    popTrlMean = popTrlMean';
    trlMean = trlMean';
    
    %% set up figure (poptrlmean)
    
    figure('color','w');
    measureNames = {'extracted (Hz)'; 'spike count (Hz)'}; %; 'SDF'};
    set(gcf, 'units', 'normalized', 'outerposition', [0.2 0.1 0.6 0.6])
    AxD = arrayfun(@(i) subplot(2,4,i,'NextPlot', 'add', 'Box', 'on', 'TickDir','out', 'FontSize', 10, 'fontweight', 'bold', ...
        'xlim', [0 nBins], 'xtick', 1:round(nBins/4):nBins, 'xticklabel', cen((0:round(nBins/4):nBins-1)+1)), 1:nArray*2);
    
    % plot all measures within an array
    arrayfun(@(i) plot(AxD(i), popTrlMean{i}, 'b','linewidth', 1), 1:nArray:nArray*2)
    arrayfun(@(i) plot(AxD(i), popTrlMean{i}, 'b','linewidth', 1), 2:nArray:nArray*2)
    arrayfun(@(i) plot(AxD(i), popTrlMean{i}, 'b','linewidth', 1), 3:nArray:nArray*2)
    arrayfun(@(i) plot(AxD(i), popTrlMean{i}, 'b','linewidth', 1), 4:nArray:nArray*2)
    
    % fix ylims
    ylims = get(AxD,'YLim');
    arrayfun(@(i) set(AxD(i),'YLim', [min(min(cell2mat(ylims(1:nArray)))) max(max(cell2mat(ylims(1:nArray))))]), 1:nArray)
    arrayfun(@(i) set(AxD(i),'YLim', [min(min(cell2mat(ylims(nArray+1:end)))) max(max(cell2mat(ylims(nArray+1:end))))]), nArray+1:2*nArray)
    
    % throw on labels and titles
    arrayfun(@(i) xlabel(AxD(i), 'sec'), 5:8)
    arrayfun(@(i) title(AxD(i), ['array ', arrayList{i}, ' (#units = ', num2str(nUnitsArray(i)), ')']), 1:nArray)
    ylabel(AxD(1), measureNames{1}), ylabel(AxD(5), measureNames{2}) %, ylabel(AxD(9), measureNames{3})
    if ismember(0, cen)
        eventXPos = find(cen==0);
    else
        eventXPos = 1;
    end
    arrayfun(@(i) line(AxD(i), [eventXPos eventXPos], [min(get(AxD(i),'YLim')) max(get(AxD(i),'YLim'))], 'color', 'black', 'linewidth', 1, 'linestyle', '--'), 1:nArray*2)
    figtitle = [nphysNameParsed(1:end-4), ...
        ' popmean (all trials) [smooth=', num2str(doSmooth)', ' win=', num2str(smoothWinSteps), ' norm=' normMethod, ' binsteps=' num2str(smoothWinSteps), ']' ];
    figtitle(strfind(figtitle,'_'))=' ';
    text(AxD(2),-0.5*nBins, 1.1*max(max(cell2mat(ylims(1:nArray)))), figtitle, 'fontweight', 'bold', 'fontsize', 12)
    print('-dtiff', '-r400', [figDir, nphysNameParsed(1:end-4), '_pop_trl_mean', paramStr])
    close
    
    %% set up heatmap figure (trlmean)
    
    figure('color','w');
    set(gcf, 'units', 'normalized', 'outerposition', [0.2 0.1 0.6 0.6])
    AxD = arrayfun(@(i) subplot(2,4,i,'NextPlot', 'add', 'Box', 'on', 'TickDir','out', 'FontSize', 10, 'fontweight', 'bold', ...
        'xlim', [0 nBins], 'xtick', 1:round(nBins/4):nBins, 'xticklabel', cen((0:round(nBins/4):nBins-1)+1)), 1:nArray*2);
    
    % plot all measures within an array
    arrayfun(@(i) imagesc(AxD(i), trlMean{i}), 1:nArray:nArray*2)
    arrayfun(@(i) imagesc(AxD(i), trlMean{i}), 2:nArray:nArray*2)
    arrayfun(@(i) imagesc(AxD(i), trlMean{i}), 3:nArray:nArray*2)
    arrayfun(@(i) imagesc(AxD(i), trlMean{i}), 4:nArray:nArray*2)
    arrayfun(@(i) colorbar(AxD(i)), 1:nArray*2)
    axis(AxD, 'tight')
    
    % fix clims
    clims = get(AxD,'CLim');
    arrayfun(@(i) set(AxD(i),'CLim', [min(min(cell2mat(clims(1:nArray)))) max(max(cell2mat(clims(1:nArray))))]), 1:nArray)
    arrayfun(@(i) set(AxD(i),'CLim', [min(min(cell2mat(clims(nArray+1:end)))) max(max(cell2mat(clims(nArray+1:end))))]), nArray+1:2*nArray)
    
    % throw on labels and titles
    arrayfun(@(i) xlabel(AxD(i), 'sec'), 5:8)
    arrayfun(@(i) title(AxD(i), ['array ', arrayList{i}]), 1:nArray)
    measureNames = {'units (extracted)', 'units (spike count)'};
    ylabel(AxD(1), measureNames{1}), ylabel(AxD(5), measureNames{2}) %, ylabel(AxD(9), measureNames{3})
    arrayfun(@(i) line(AxD(i), [eventXPos eventXPos], [min(get(AxD(i),'YLim')) max(get(AxD(i),'YLim'))], 'color', 'black', 'linewidth', 1, 'linestyle', '--'), 1:nArray*2)
    figtitle = [nphysNameParsed(1:end-4), ...
        ' mean (all trials) [smooth=', num2str(doSmooth)', ' win=', num2str(smoothWinSteps), ' norm=' normMethod, ' binsteps=' num2str(smoothWinSteps), ']' ];
    figtitle(strfind(figtitle,'_'))=' ';
    text(AxD(2),-0.5*nBins, 1.3*(nUnits/nArray), figtitle, 'fontweight', 'bold', 'fontsize', 12)
    print('-dtiff', '-r400', [figDir, nphysNameParsed(1:end-4), '_htmp_trl_mean', paramStr])
    close
    
    %% export arrays for indexing by unit and by trial type
    
    save([outDir, 'spikeData_', nphysNameParsed(1:end-4)], 'alignedBinnedSpikes', 'params', 'spikeCount', '-v7.3')
    save([outDir, 'spikeInfo_', nphysNameParsed(1:end-4)], 'spikeInfo', '-v7.3')
    save([outDir, 'trlInfo_', nphysNameParsed(1:end-4)], 'trlInfo', '-v7.3')
    
    clear alignedBinnedSpikes alignedSpikes spikeCount spikesSDF arraySpikeCount arraySpikes Coder aTS
    
end

end

