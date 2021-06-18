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
% dsAllSpikes: #neurons (from nip1 and nip2 together) by #binsteps of
% spiking data covering the whole session.
%
% dsSpikeTimes: 1 x #binsteps clocktimes for each column of dsAllSpikes.

% dsAllEvents: 1 x duration of session long vector of 0s with 1s =
% fixation, 2s = stim, 3s = choice, 4s = outcome.
%
% spikeInfo: table (it's just the Neurons variable from the
% #YYYYMMDD_nip#.mat files, but labeled more clearly) for a given session,
% monkey, and NIP. See spikeInfoCell for cell version.
%
% trlInfo: table from Coder and Coder.BHVOut variables, labeled more
% clearly, for a given session, monkey, and NIP. See trlInfoVarNames for
% more detailed description. See trlInfoCell for cell version.
%
% alignedBinnedSpikes: my version, #neurons x #trials x #bins for a given
% monkey and session, according to params.
%
% params: params for the above variables (lockEvent and winBounds apply to
% alignedBinnedSpikes only and are N/A for dsAllSpikes).
%
%% TO DO

% 1. confirm about msec assumption on timings and spikes confirm about
% sampling. CHECK!!! paper says 1 kHz (1000/s so 0.001 place but this seems
% to be in 10 kHz. Assuming that this is in fractions of a second, based on
% number of decimal places in aTS and event times.

%% housekeeping

clearvars;
close all

%% input params

% data specs
% monkeyList              = {'v', 'w'};
nipList                 = [1, 2];
lockEvent               = 'fixation';       % fixation, stim, choice, reward
winBounds               = [0 2.8];          % in sec around lockEvent

% data processing params
normMethod              = 'none';           % 2norm, zscore, maxscale, none
doSmooth                = true;             % true or false
binSteps                = 10;               % in sample points, can be any number < dimensions of extracted window
initBinSize             = 0.001;            % in msec; a function of original data sampling rate, also an assumption
smoothWidth             = 100;              % in init sample points (for width in msec multiply by initBinSize)

% derived from data processing params
binWidth                = binSteps * initBinSize;
binSizeStr              = num2str(initBinSize);
numDP                   = numel(binSizeStr(strfind(binSizeStr,'.')+1:end));

% overwrite defaults based on inputs
if exist('params','var')
    assignParams(who,params);
end

% package up params for reference
% params.monkeyList       = monkeyList;
params.nipList          = nipList;
params.lockEvent        = lockEvent;
params.winBounds        = winBounds;
params.normMethod       = normMethod;
params.doSmooth         = doSmooth;
params.binSteps         = binSteps;
params.initBinSize      = initBinSize;
params.smoothWidth      = smoothWidth;
params.binWidth         = binWidth;
params.numDP            = numDP;

tmpParams = params;

paramStr = ['_smth_', num2str(doSmooth)', '_win_', num2str(binSteps), '_norm_' normMethod, '_bnstp_' num2str(binSteps)];

% define directories
coderDir                = '~/Dropbox/costa_learning/data/coder_files/';
nphysDir                = '~/Dropbox/costa_learning/data/neurophys_data/';
figDir                  = '~/Dropbox/costa_learning/figures/';
outDir                  = '~/Dropbox/costa_learning/reformatted_data/';

%% set up

cd(coderDir)
coderFiles = dir('*1.mat');
cd(nphysDir)
nphysFiles = dir('*1.mat');

% name parsing depends on length of these names all being consistent
coderNameLength = arrayfun(@(x) length(coderFiles(x).name), 1:length(coderFiles));
assert(numel(unique(coderNameLength))==1)
nphysNameLength = arrayfun(@(x) length(nphysFiles(x).name), 1:length(nphysFiles));
assert(numel(unique(nphysNameLength))==1)

trlInfoVarNames = ...
    {'trls_since_nov_stim', ...         % trlssincenov: #trials since novel stimulus, ...
    'nov_stim_idx', ...                 % picid: (when & where the novel image appeared)
    'nov_stim_loc', ...                 % dropin_loc (novel trial' location )
    'nov_stim_rwd_prob', ...            % dropinrewprob (novel trial' reward probability )
    'stimID', ...                       % stim: stimID (it's an actual value but TO DO: find out what the actual value means)
    'reward_probs', ...                 % rrprob_aval: reward probability
    'choice_idx', ...                   % choices: which stim was chosen, will be 1, 2, or 3 (it's an index)
    'chosen_stim_ud', ...               % orientation: orientation of the (chosen) stimulus triangle (up or down)
    'chosen_stim_dir', ...              % direction: direction of the chosen stimulus (1 - 6)
    'chosen_stim', ...                  % chosenstim: Coder.stim(Coder.choices) = Coder.chosenstim (or, stimID(choice_idx) = chosen_stim)
    'chose_nov_stim', ...               % chosenovel (whether chose the novel image)
    'reward', ...                       % reward: got a reward or not
    'event_times', ...                  % 1=fixlock (fixation), 2=stimlock (stimulus on), 3=choice, 4=juicelock (outcome)
    'aligned_event_times'};             % the above but with lockEventTime subtracted

spikeInfoVarNames = {'monkey', 'array', 'date', 'nip', 'electrode', 'unit', 'port'};

% create half-gaussian kernel
w = gausswin(2*(smoothWidth/binSteps)); w(1:(smoothWidth/binSteps)) = [];

%% Loop through neurophys and behavioral files to extract all spike info we
% might care about and put all event info we might care about into a
% helpful table. For my own edification extract spikes as well

for iCoder = 1:length(coderFiles)
    
    alignedBinnedSpikes     = [];
    spikeInfo               = [];
    trlInfo                 = {};
    spikeCountAll           = [];
    allEvents               = [];
    dsAllSpikes             = [];
    
    coderfName = coderFiles(iCoder).name;
    brks = strfind(coderfName, '_');
    monkey = coderfName(brks(1)+1);
    sessionDate = coderfName(brks(2)+1:brks(3)-1);
    
    % update params for given monkey and session
    tmpParams.monkey = monkey;
    tmpParams.sessionDate = sessionDate;
    params = [fieldnames(tmpParams) struct2cell(tmpParams)];

    % get latest spike of the 2 nips
    load([nphysDir, monkey, sessionDate, '_nip1.mat'], 'aTS', 'Coder')
    T1 = aTS; C1 = Coder; clear aTS Coder
    load([nphysDir, monkey, sessionDate, '_nip2.mat'], 'aTS', 'Coder')
    T2 = aTS; C2 = Coder; clear aTS Coder
    
    lastSpike = round(max([max(cellfun(@max, T1)), max(cellfun(@max, T2))]), numDP, 'decimals');
    allPossSpikeTimes = round([0 : initBinSize : lastSpike], numDP, 'decimals');
    
    while mod(size(allPossSpikeTimes,2), binSteps) ~= 0 % need windowWidth to be evenly divisible by specified binWidth
        allPossSpikeTimes = [allPossSpikeTimes, (allPossSpikeTimes(end) + 10^(-numDP))];
    end
    
    nVecBins = size(allPossSpikeTimes,2) / binSteps;
    eventsN1 = [C1.fixlock, C1.stimlock, C1.stimlock + C1.srt./1000, C1.juicelock];
    eventsN2 = [C2.fixlock, C2.stimlock, C2.stimlock + C2.srt./1000, C2.juicelock];
    allLag = eventsN2 - eventsN1;
    
    clear T1 T2 C1 C2
    
    % nip2 is same session same day same monkey just different machine with
    % a SLIGHT difference in clock (~10ms) so we gotta pull its spikes
    % separately and then stick em together
    for iNip = [1 2]
        
        nipNum = iNip;
        nphysNameParsed = [monkey, sessionDate, '_nip', num2str(nipNum), '.mat'];
        load([nphysDir, nphysNameParsed], 'aTS', 'Bin', 'Coder', 'Neurons', 'spikeCount')
        
        % make sure we are extracting non error trials only
        assert(isequal(Coder.validtrls, find(Coder.BHVerror==0)))
        
        % make sure trials are not overlapping (end of previous trial must be before start of current)
        minDurBtwnTrls = min(Coder.fixlock(2:end) - (Coder.fixlock(1:end-1)));
        assert(range([0 winBounds(2)]) <= minDurBtwnTrls)
        
        % extract trial event info
        eventTimes = [Coder.fixlock, Coder.stimlock, Coder.stimlock + Coder.srt./1000, Coder.juicelock];
        assert(all(all(diff(eventTimes, [],2)>=0))) % make sure these are happening in the correct order
        
        % there is a difference in absolute event times between nips, so
        % adjust nip2 accordingly - assumes nip2 is later?
        if nipNum == 2
            % put nip2 in clock of nip1
            lagSec = mean(allLag,2);
            % adjust nip2 event times
            eventTimes = eventTimes - lagSec;
        end
        
        % round everything to reasonable precision
        % how this is rounded changes how precise our bin size can be - not
        % sure if this is artifact or real
        eventTimes = round(eventTimes, numDP, 'decimals');
        
        switch lockEvent
            case 'fixation'
                lockEventTime = eventTimes(:, 1); % Coder.fixlock;
            case 'stim'
                lockEventTime = eventTimes(:, 2); % Coder.stimlock;
            case 'choice'
                lockEventTime = eventTimes(:, 3); % Coder.stimlock + Coder.srt./1000;
            case 'reward'
                lockEventTime = eventTimes(:, 4); % Coder.juicelock; % if aligned on choice, this will also be aligned
        end
        
        eventTimesAligned = round(eventTimes - lockEventTime, numDP, 'decimals');
        
        % prep for binning
        nTrls = length(Coder.validtrls);
        nUnits = size(Neurons,1);
        nBins = round((range(winBounds) + 3 * binWidth) ./ binWidth);
        
        % get outputs for this session
        rrprob_aval = Coder.BHVout(:,15:17);
        nov_stim_loc = Coder.BHVout(:,26);
        nov_stim_rwd_prob = Coder.BHVout(:,27);
        chose_nov_stim = Coder.BHVout(:,31);
        picID = arrayfun(@(iTrl) find(Coder.picid(iTrl,:)), 1:nTrls, 'UniformOutput', 0)';
        picID(cellfun(@isempty,picID))={0};
        picID = cell2mat(picID);
        
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
        
        % bounds for collecting spikes - with a little extra so winBounds are
        % actually win centers, as in Bin.cen
        iStart = winBounds(1) - (10 * binWidth) + initBinSize;
        iStop = winBounds(2) + (10 * binWidth);
        
        % for plotting
        edges = (iStart:(iStop - iStart)/nBins:iStop)';
        cen = edges(1:end-1) + (diff(edges) ./ 2);
        nBins = nBins - 3;
        
        % align ith unit's spikes around lockEvent for all trials
        winTimes = [lockEventTime+iStart, lockEventTime+iStop]; % bounds for all trials for ith unit
        
        % clock indices for winTimes
        allTrlTS = cell2mat(arrayfun(@(iTrl) winTimes(iTrl,1):initBinSize:winTimes(iTrl,2), 1:nTrls, 'un', 0)');
        
        alignedBinnedSpikes_curr = NaN(nUnits, nTrls, nBins);
        
        % also reshape all trials into one long continuous vector
        eventTimesVec = reshape(eventTimes',1,numel(eventTimes));
        allEvents_curr = zeros(size(allPossSpikeTimes));
        
        % must match
        if ~all(ismember(round(eventTimesVec,numDP,'decimals'), allPossSpikeTimes))
            disp('precision issue')
            keyboard
        end
        
        if ~isequal(numel(round(eventTimesVec, numDP, 'decimals')), sum(ismember(allPossSpikeTimes,round(eventTimesVec, numDP, 'decimals'))))
            keyboard
        end
        
        % create event labels: 1 = fixation, 2 = stim, 3 = choice, 4 = outcome
        for iEvent = 1:size(eventTimes,2)
            allEvents_curr(ismember(allPossSpikeTimes,eventTimes(:,iEvent))) = iEvent;
        end
        
        if ~isequal(nTrls, sum(allEvents_curr==1))
            keyboard
        end
        
        % initialize big long vector of unstacked trials
        allSpikes_curr = zeros(nUnits, nVecBins);
        
        tic
        for nn = 1:nUnits % For each unit in file
            
            % pull all of a given unit's spike timestamps
            aTSUnit = aTS{nn};
            allSpikeTS = round(aTSUnit, numDP, 'decimals'); 
            
            % get timestamps for spikes within bounds (winTimes)
            winSpikeTS = arrayfun(@(iTrl) allSpikeTS ...
                (allSpikeTS > winTimes(iTrl,1) ...
                & allSpikeTS <= winTimes(iTrl,2)), ...
                1:nTrls, 'UniformOutput', false)';
            
            % count all spikes within winTimes for each trial for ith unit
            N = cell2mat(arrayfun(@(iTrl) ...
                histcounts(winSpikeTS{iTrl}, winTimes(iTrl,1):initBinSize:winTimes(iTrl,2)), ...
                1:nTrls, 'UniformOutput', false)'); % trials x #made-up sample points for ith unit
            
            % grab all spikes in alternative continuous format for RNNs
            spikeVec = double(ismember(allPossSpikeTimes, allSpikeTS));
            
            % normalize
            switch normMethod
                case '2norm'
                    toNorm = mean(N,2)~=0;
                    Nnorm = zeros(size(N));
                    Nnorm(toNorm,:) = normalize(N(toNorm,:),2,'norm'); % feature values proportionate to each other (euclidean norm) - each trial will be proportionate to each other
                    N = Nnorm;
                case 'zscore'
                    N = zscore(N, 0, 2);
                case 'maxscale'
            end
            
            % calculate FR with sliding window of 100ms (#trials x #bins for ith unit)
            padding = zeros(1, 0.5 * 10 * binSteps); % only for vector format since there is no extra data to pad with
            % frN = binData(N, struct('sliding', true, 'binWidth', 10 * binSteps, 'step', 1)); % downsample(N', binSteps)'
            % frSpikeVec = binData([padding spikeVec padding(1:end-1)], struct('sliding', true, 'binWidth', 10 * binSteps, 'step', 1)); % downsample(spikeVec, binSteps);
            frN = movmean(N, 10 * binSteps, 2);
            frSpikeVec = movmean([padding spikeVec padding], 10 * binSteps, 2);
            
            % trim off the extra bins at start and end
            overhang = round(-1 * (winBounds(2) - iStop) ./ initBinSize);
            frN = frN(:, overhang : end - overhang);
            frSpikeVec = frSpikeVec(:, length(padding) : end - length(padding) - 1);
            
            if length(frSpikeVec) ~= length(spikeVec)
                keyboard
            end
            
            % downsample
            % dsN = binData(frN, struct('sliding', false, 'binWidth', binSteps)); % downsample(N', binSteps)'
            % dsSpikeVec = binData(frSpikeVec, struct('sliding', false, 'binWidth', binSteps)); % downsample(spikeVec, binSteps);
            dsN = downsample(frN', binSteps)';
            dsSpikeVec = downsample(frSpikeVec, binSteps);
            
            %  smooth by kernel as wide as their binning window
            if doSmooth
                % smoothing with a half-gaussian
                dsN = cell2mat(arrayfun(@(n) conv(dsN(n, :), w, 'same'), 1:size(dsN,1), 'un', 0)');
                dsSpikeVec = conv(dsSpikeVec, w, 'same');
            end
            
            alignedBinnedSpikes_curr(nn, :, :) = dsN;
            allSpikes_curr(nn, :) = dsSpikeVec;
        end
        
        toc
        
        cen = cen(cen>=winBounds(1) & cen<=winBounds(2));
        
        % concatenate nips along #neurons dimension (#trials and #bins
        % should be the same)
        alignedBinnedSpikes = cat(1,alignedBinnedSpikes,alignedBinnedSpikes_curr);
        spikeCountAll = cat(1, spikeCountAll, spikeCount);
        
        spikeInfo = [spikeInfo; spikeInfo_curr];
        trlInfo{iNip} = trlInfo_curr;
        
        if nipNum == 1 % save for comparison for when you get common spiketimes
            allEvents_n1 = allEvents_curr;
            allSpikes_n1 = allSpikes_curr;
            
        elseif nipNum == 2 % get common spiketimes for both NIPs
            allEvents_n2 = allEvents_curr;
            allSpikes_n2 = allSpikes_curr;
            
            % even adjusting for nip2 lag from nip1, rounding errors make
            % it so that event flags for nip2 are ahead of nip1 by 1 msec
            % usually. idk how to use this as a diagnostic or if it
            % matters.
            allEvents = allEvents_n1;
            dsAllSpikes = [allSpikes_n1; allSpikes_n2];
        end
    end
    
    
    % downsample relevant bits
    dsSpikeTimes = downsample(allPossSpikeTimes, binSteps);
    dsAllEvents = zeros(1, nVecBins);
    for iEvent = 1:size(eventTimes,2)
        inds = find(allEvents == iEvent)'; % tells us where in allSpikes each fixation is and gives us common inds for T and allSpikes
        TS = allPossSpikeTimes(inds);
        % for each element of downsampled allPossSpikeTimes aka tDataAll, get ind of
        % closest match between fixTS and tDataAll to make downsample eventFlags
        dsInds = arrayfun(@(i) find(abs(dsSpikeTimes - TS(i)) == min(abs(dsSpikeTimes - TS(i))), 1), 1:nTrls);
        dsAllEvents(dsInds) = iEvent;
        
    end
    
    % sanity check trlInfo
    assert(isequal(trlInfo{1}(:, 1:12), trlInfo{2}(:, 1:12)))
    trlInfo = trlInfo{1};
    
    trlInfoCell = [trlInfo.Properties.VariableNames; table2cell(trlInfo)];
    spikeInfoCell = [spikeInfo.Properties.VariableNames; table2cell(spikeInfo)];
    
    % update nUnits
    nUnits = size(alignedBinnedSpikes, 1);
    
    %% extract variables for plotting by array
    
    arrayList = unique(spikeInfo.array);
    nArray = length(arrayList);
    
    % extract metrics for each array
    popTrlMean = cell(1, nArray);
    trlMean = cell(1, nArray);
    nUnitsArray = NaN(1,nArray);
    for aa = 1:nArray
        inArray = strcmp(spikeInfo.array,arrayList{aa});
        nUnitsArray(aa) = sum(inArray);
        arraySpikes = alignedBinnedSpikes(inArray,:,:);
        % convert to Hz by multiplying mean spikes/bin by (1 / binWidth) bins / sec
        popTrlMean{1, aa} = squeeze(mean(arraySpikes, [1 2])) ./ binWidth;
        trlMean{1, aa} = squeeze(mean(arraySpikes, 2)) ./ binWidth;
    end
    
    popTrlMean = popTrlMean';
    trlMean = trlMean';
    
    %% set up figure (poptrlmean)
    
    figTitleStr = ['[smooth=', num2str(doSmooth)', ' win=', num2str(binSteps), ' norm=' normMethod, ' binsteps=' num2str(binSteps), ']'];
    if ismember(0, cen)
        eventXPos = find(cen==0);
    else
        eventXPos = 1;
    end
    
    tickLabels = linspace(winBounds(1),winBounds(2),7);
    tickLabels(1) = [];
    figure('color','w');
    measureNames = {'extracted (Hz)'}; %; 'SDF'};
    set(gcf, 'units', 'normalized', 'outerposition', [0.2 0.1 0.6 0.6])
    AxD = arrayfun(@(i) subplot(2,4,i,'NextPlot', 'add', 'Box', 'on', 'TickDir','out', 'FontSize', 10, 'fontweight', 'bold', ...
        'xlim', [0 nBins], 'xtick', tickLabels ./ binWidth, 'xticklabel',tickLabels), 1:nArray);
    
    % plot all measures within an array
    arrayfun(@(i) plot(AxD(i), popTrlMean{i}, 'b','linewidth', 1), 1:nArray)
    
    % fix ylims, throw on labels and titles
    ylims = get(AxD(nUnitsArray >= 10),'YLim');
    arrayfun(@(i) set(AxD(i),'YLim', [min(min(cell2mat(ylims(1:end)))) max(max(cell2mat(ylims(1:end))))]), 1:nArray)
    arrayfun(@(i) xlabel(AxD(i), 'sec'), 1:nArray)
    arrayfun(@(i) title(AxD(i), ['array ', arrayList{i}, ' (#units = ', num2str(nUnitsArray(i)), ')']), 1:nArray)
    ylabel(AxD(1), measureNames{1}), ylabel(AxD(5), measureNames{1})
    arrayfun(@(i) line(AxD(i), [eventXPos eventXPos], [min(get(AxD(i),'YLim')) max(get(AxD(i),'YLim'))], 'color', 'black', 'linewidth', 1, 'linestyle', '--'), 1:nArray)
    figtitle = [monkey, sessionDate, ' popmean (all trials) ', figTitleStr];
    figtitle(strfind(figtitle,'_'))=' ';
    text(AxD(2),-0.5*nBins, 1.12*max(max(cell2mat(ylims(1:end)))), figtitle, 'fontweight', 'bold', 'fontsize', 12)
    print('-dtiff', '-r400', [figDir, [monkey, sessionDate], '_pop_trl_mean', paramStr])
    close
    
    %% set up heatmap figure (trlmean)
    
    figure('color','w');
    set(gcf, 'units', 'normalized', 'outerposition', [0.2 0.1 0.6 0.6])
    AxD = arrayfun(@(i) subplot(2,4,i,'NextPlot', 'add', 'Box', 'on', 'TickDir','out', 'FontSize', 10, 'fontweight', 'bold', ...
        'xlim', [0 nBins], 'xtick', tickLabels ./ binWidth, 'xticklabel', tickLabels), 1:nArray);
    
    % plot all measures within an array
    arrayfun(@(i) imagesc(AxD(i), trlMean{i}), 1:nArray)
    arrayfun(@(i) colorbar(AxD(i)), 1:nArray)
    axis(AxD, 'tight')
    
    % fix clims, throw on labels and titles
    clims = get(AxD,'CLim');
    arrayfun(@(i) set(AxD(i),'CLim', [min(min(cell2mat(clims(1:nArray)))) max(max(cell2mat(clims(1:nArray))))]), 1:nArray)
    arrayfun(@(i) xlabel(AxD(i), 'sec'), 5:8)
    arrayfun(@(i) title(AxD(i), ['array ', arrayList{i}, ' (#units = ', num2str(nUnitsArray(i)), ')']), 1:nArray)
    measureNames = {'units (extracted)', 'units (spike count)'};
    ylabel(AxD(1), measureNames{1}), ylabel(AxD(5), measureNames{1})
    arrayfun(@(i) line(AxD(i), [eventXPos eventXPos], [min(get(AxD(i),'YLim')) max(get(AxD(i),'YLim'))], 'color', 'black', 'linewidth', 1, 'linestyle', '--'), 1:nArray)
    figtitle = [monkey, sessionDate, ' mean (all trials) ', figTitleStr ];
    figtitle(strfind(figtitle,'_'))=' ';
    text(AxD(2),-0.5*nBins, 1.3*(nUnits/nArray), figtitle, 'fontweight', 'bold', 'fontsize', 12)
    print('-dtiff', '-r400', [figDir, [monkey, sessionDate], '_htmp_trl_mean', paramStr])
    close
    
    %% export arrays for indexing by unit and by trial type
    
    save([outDir,  monkey, sessionDate, '_spikesCont'],  'dsAllSpikes',   'params', '-v7.3')
    save([outDir, monkey, sessionDate, '_spikesAligned'],  'alignedBinnedSpikes', 'params', '-v7.3')
    save([outDir, monkey, sessionDate, '_meta'],  'dsSpikeTimes', 'dsAllEvents', 'spikeInfo', 'trlInfo', 'spikeInfoCell', 'trlInfoCell', '-v7.3')
    
    clear allSpikes allSpikes_n1 allSpikes_n2 allSpikes_curr spikeCount alignedBinnedSpikes arraySpikes allEventFlags allEvent_flags_curr allEvent_flags_n1 allEventFlags_n2 allPossSpikeTimes arraySpikeCount  Coder aTS
    
end

end

