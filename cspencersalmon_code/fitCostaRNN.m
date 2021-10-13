function RNN = fitCostaRNN(RNNname, ...
    allSpikes, allPossTS, allEvents, spikeInfo, T, arrayRgns, params)
rng(42)

%% TO DO

% decide what networks stats to save
% decide what other params to save
% decide whether to train on a trial or a set or all trials in a set
% decide what Js to save and what Js to feed in to each link in the chain -
% do you start with a new J for every set or???
% do you re-initialize off the first datapoint for every trial or???
%  DOES IT MATTER THAT WE ARE INITIALIZING DIFFERENT FROZEN WHITE NOISE
% INPUTS EVERY TIME WE START TRAINING A CHUnk?

%% data meta
% datadir         = '~/Dropbox (BrAINY Crew)/costa_learning/reformatted_data/';
mouseVer        = '';

%% data parameters
dtData          = 0.010;                % time step (in s) of the training data
dtFactor        = 20;                   % number of interpolation steps for RNN
doSmooth        = false;
smoothWidth     = 0.15;                 % in seconds, width of gaussian kernel if doSmooth == true
doSoftNorm      = false;
normByRegion    = false;                % normalize activity by region or globally
rmvOutliers     = true;

%% RNN parameters
g               = 1.5;                  % instability (chaos); g<1=damped, g>1=chaotic
tauRNN          = 0.001;                % decay costant of RNN units ?in msec
tauWN           = 0.1;                  % decay constant on filtered white noise inputs
ampInWN         = 0.001;                % input amplitude of filtered white noise

%% training params
alpha           = 1;                    % overall learning rate for regularizer
nonlinearity    = @tanh;                % inline function for nonlinearity
nRunTrain       = 1000;
resetPoints     = 1;                    % default to only set initial state at time 1
trainFromPrev   = false;                % assume you're starting from beginning, but if not, feed in previous J

%% output options
plotStatus      = true;
saveMdl         = true;


%% RNN

% overwrite defaults based on inputs
if exist('params','var')
    assignParams(who,params);
end

% define output directories
rnnDir          = ['~/Dropbox (BrAINY Crew)/costa_learning/models/', mouseVer, filesep];
rnnSubDir       = [rnnDir, RNNname(strfind(RNNname, '_') + 1 : end), filesep];
rnnFigDir       = ['~/Dropbox (BrAINY Crew)/costa_learning/figures/', mouseVer, filesep];

if ~isfolder(rnnSubDir)
    mkdir(rnnSubDir)
end

% set up final params
dtRNN           = dtData / dtFactor;    % time step (in s) for integration
ampWN           = sqrt( tauWN / dtRNN );
nRunFree        = ceil(0.01 * nRunTrain);
nRunTot         = nRunTrain + nRunFree;   % idk according to CURBD

% preprocess targets by smoothing, normalizing, re-scaling, and outlier removing
targets = allSpikes;

if doSmooth
    targets = smoothdata(targets, 2, 'gaussian', smoothWidth / dtData); % convert smoothing kernel from msec to #bins);
end

% this will soft normalize a la Churchland papers
if doSoftNorm
    normfac = range(targets, 2); % + (dtData * 10); % normalization factor = firing rate range + alpha
    targets = targets ./ normfac;
end

if normByRegion
    arrayList = unique(spikeInfo.array);
    nArray = length(arrayList);
    
    for aa = 1:nArray
        inArray = strcmp(spikeInfo.array,arrayList{aa});
        
        arraySpikes = targets(inArray, :);
        targets(inArray,:) = arraySpikes ./ max(max(arraySpikes));
    end
else
    targets = targets ./ max(max(targets));
end

if rmvOutliers
    figure('color','w');
    set(gcf, 'units', 'normalized', 'outerposition', [0.05 0.1 0.9 0.6])
     AxD = arrayfun(@(i) subplot(1,2,i,'NextPlot', 'add', 'Box', 'on', ...
         'TickDir','out', 'FontSize', 10, 'fontweight', 'bold'), 1:2);
   
    subplot(1, 2, 1), histogram(mean(targets, 2)), title('mean target FRs all units w/ outliers')
    outliers = isoutlier(mean(targets, 2), 'percentiles', [0.5 99.5]);
    targets = targets(~outliers, :);
    spikeInfo = spikeInfo(~outliers, :);
    subplot(1, 2, 2), histogram(mean(targets, 2)), title(['mean target FRs all units minus ', num2str(sum(outliers)), ' outliers'])
    
    oldxlim = cell2mat(get(AxD, 'xlim'));
    newxmin = min(oldxlim(:)); newxmax = max(oldxlim(:));
    arrayfun(@(i) set(AxD(i), 'xlim', [newxmin newxmax]), 1:2)
    tmpFigName = RNNname;
    tmpFigName(strfind(tmpFigName, '_')) = ' ';
    text(AxD(2), -0.3 * (newxmax - newxmin), 1.05 * max(get(AxD(2), 'ylim')), tmpFigName, 'fontweight','bold', 'fontsize', 13)
    print('-dtiff', '-r400', [rnnFigDir, 'targets_outlier_comparison_', RNNname])
    close
    
    % update indexing vectors
    % arrayfun(@(iRgn) arrayRgns{iRgn, 3}((find(outliers))=[]), 3:size(arrayRgns,1), 'un', 0)
    for iRgn = 1 : size(arrayRgns, 1)
        arrayRgns{iRgn, 3}(outliers) = [];
    end
    
end

% housekeeping
if any(isnan(targets(:)))
    keyboard
end

if plotStatus
    figure, subplot(1,2,1), imagesc(allSpikes), title('non-normed rates'), colorbar, colormap jet, subplot(1,2,2), imagesc(targets), colorbar, title('target rates')
    if ~isfolder([rnnFigDir, 'targets_', RNNname])
        mkdir([rnnFigDir, 'targets_', RNNname])
    end
    print('-dtiff', '-r400', [rnnFigDir, 'targets_', RNNname])
    close
end

% set up for training
nUnits = size(targets, 1);
nLearn = nUnits; % number of learning steps

% if the RNN is bigger than training neurons, pick the ones to target (??)
learnList = 1 : nLearn;% randperm(nUnits);
iTarget = learnList(1:nLearn);
iNonTarget = learnList(nLearn:end);

% sanity check match in #trials
assert(isequal(sum(allEvents == 1), height(T)))
nTrls = height(T);

% pull trial starts
fixOnInds = [find(allEvents == 1), size(targets, 2)];

% get block/set structure (s sets of j trials each)
firstTrlInd = find(T.trls_since_nov_stim == 0);
lastTrlInd = find(T.trls_since_nov_stim  == 0) - 1;
if lastTrlInd(1) ~= 0, keyboard, end
if firstTrlInd(1) ~= 1, keyboard, end
firstTrlInd(1) = [];
lastTrlInd(1:2) = [];
lastTrlInd = [lastTrlInd; height(T)]; % can't have a 0 ind
nTrlsPerSet = T.trls_since_nov_stim(lastTrlInd) + 1; % 10 <= j <= 30 according to paper
nSets = length(nTrlsPerSet); % s <= 32 according to paper
setID = [0; repelem(1:nSets, nTrlsPerSet)'];

% initialize outputs
stdData = zeros(1,nTrls);
JTrls = NaN(nUnits, nUnits, nTrls);
  
try
    prevMdls = dir([rnnSubDir, RNNname, '_set*_trial*.mat']);
    allTrialIDs = unique(arrayfun(@(i) ...
        str2double(prevMdls(i).name(strfind(prevMdls(i).name,'trial') + 5 : end - 4)), ...
        1 : length(prevMdls)));
    % prevMdl = load([rnnSubDir prevMdls(allTrialIDs == max(allTrialIDs)).name]);
    prevMdl = load([rnnSubDir, dir([rnnSubDir, RNNname, ...
        '_set*_trial', num2str(max(allTrialIDs)), '.mat']).name]);
    prevJ = prevMdl.RNN.mdl.J;
    prevJ0 = prevMdl.RNN.mdl.J0;
    prevTrl = prevMdl.RNN.mdl.iTrl;
    if trainFromPrev
        startTrl = prevMdl.RNN.mdl.iTrl + 1; % max(allTrialIDs) + 1;
    else
        startTrl = 1;
        prevTrl = 0;
    end
catch
    startTrl = 1;
    prevTrl = NaN;
end


for iTrl = startTrl : nTrls % - 1 or nSets - 1

    fprintf('\n')
    
    disp([RNNname, ': training trial # ', num2str(iTrl), '.'])
    
    iStart = fixOnInds(iTrl); % start of trial
    iStop = fixOnInds(iTrl + 1) - 1; % right before start of next trial
    currTargets = targets(:, iStart:iStop);
    
    tData = allPossTS(iStart:iStop); % timeVec for current data
    tRNN = tData(1) : dtRNN : tData(end); % timevec for RNN
    
    % set up white noise inputs (from CURBD)
    iWN = ampWN * randn( nUnits, length(tRNN) );
    inputWN = ones(nUnits, length(tRNN));
    
    for tt = 2 : length(tRNN)
        inputWN(:, tt) = iWN(:, tt) + (inputWN(:, tt - 1) - iWN(:, tt)) * exp( -(dtRNN / tauWN) );
    end
    
    inputWN = ampInWN * inputWN;
    
    % sidebar to save inputWN separately in the hopes of making loading
    % RNNs faster
    if ~isfolder([rnnSubDir, 'WN/'])
        mkdir([rnnSubDir, 'WN/'])
    end
    
    inputWNDataFnm = [rnnSubDir, 'WN/', RNNname, '_inputWN_trl', num2str(iTrl), '.mat'];
    
    if saveMdl
        save(inputWNDataFnm, 'inputWN', '-v7.3')
    end
        
    % initialize DI matrix J
    if trainFromPrev && iTrl == startTrl && exist('prevJ', 'var')
        J = prevJ;
    else
        if iTrl == 1
            J = g * randn(nUnits,nUnits) / sqrt(nUnits);
        else
            J = squeeze(JTrls(:, :, iTrl - 1));
        end
    end
    
    J0 = J; % TO DO: ADD CHECK THAT J0(t) = J(t-1) aka prevJ
    
    % get standard deviation of entire data that we are looking at
    stdData(iTrl)  = std(reshape(currTargets(iTarget,:), length(iTarget)*length(tData), 1));
    
    % get indices for each sample of model data for getting pVar
    iModelSample = zeros(length(tData), 1);
    for i = 1:length(tData)
        [~, iModelSample(i)] = min(abs(tData(i) - tRNN));
    end
    
    % initialize some others
    R = zeros(nUnits, length(tRNN)); % rate matrix - firing rates of neurons
    chi2 = zeros(1, nRunTot);
    pVars = zeros(1, nRunTot);
    JR = zeros(nUnits, length(tRNN)); % z(t) for the output readout unit
    
    % initialize learning update matrix (see Sussillo and Abbot, 2009)
    PJ = alpha * eye(nUnits); % dim are pN x pN where p=fraction of neurons to modify - here it's all of them
    
    if plotStatus
        f = figure('Position',[100 100 1800 600]);
    end
    
    % tic
    
    %% training
    
    % loop through training runs
    for nRun = 1 : nRunTot
        % set initial condition to match target data
        H = currTargets(:, 1);
        
        % convert to currents through nonlinearity
        R(:, 1) = nonlinearity(H);
        
        tLearn = 0; % keeps track of current time
        iLearn = 1; % keeps track of last data point learned
        JLearn = [];
        
        for tt = 2 : length(tRNN) % why start from 2?
            tLearn = tLearn + dtRNN;
            
            % check if the current index is a reset point. Typically this won't
            % be used, but it's an option for concatenating multi-trial data
            if ismember(tt, resetPoints)
                H = currTargets(:, floor(tt / dtFactor) + 1);
            end
            
            
            % compute next RNN step
            R(:, tt) = nonlinearity(H); %1./(1+exp(-H));
            
            % debug
%             if std(R(:, tt)) > 3 * mean(std(R(:, 2:tt)))
%                 keyboard
%             end
            
            JR(:, tt) = J * R(:, tt) + inputWN(:, tt);  % zi(t)=sum (Jij rj) over j
            H = H + dtRNN * (-H + JR(:, tt)) / tauRNN; % update activity
            
            % check if the RNN time coincides with a data point to update J
            if tLearn >= dtData
                tLearn = 0;
                
                % error signal --> z(t)-f(t), where f(t) = target function
                % if currTargets are treated as currents, compare JR
                % if currTargets treated as smoothed rates, compare RNN
                error = R(1:nUnits, tt) - currTargets(1:nUnits, iLearn);
                
                
                if norm(error) > 10
                    keyboard
                end
                
                % update chi2 using this error
                chi2(nRun) = chi2(nRun) + mean(error.^2);
                
                % update learning index
                iLearn = iLearn+1;
                if (nRun <= nRunTrain)
                    
                    % update terms for training runs
                    k = PJ * R(iTarget, tt); % N x 1
                    rPr = R(iTarget, tt)' * k; % scalar; inverse cross correlation of network firing rates
                    c = 1 / (1 + rPr); % learning rate
                    PJ = PJ - c * (k * k');
                    J(1:nUnits, iTarget) = J(1:nUnits, iTarget) - c * error(1:nUnits, :) * k';
                    JLearn(:, :, iLearn - 1) = J; % for each learning step of a given run
                end
                
            end
        end
        
        rModelSample = R(iTarget, iModelSample);
        
        % compute variance explained of activity by units
        froNorm(nRun) = norm(currTargets(iTarget,:) - rModelSample, 'fro' );
        sqrtNTstd(nRun) = ( sqrt(length(iTarget) * length(tData)) * stdData(iTrl));
        
        pVar = 1 - ( norm(currTargets(iTarget,:) - rModelSample, 'fro' ) / ( sqrt(length(iTarget) * length(tData)) * stdData(iTrl)) ).^2;
        
        if pVar < 0
            keyboard
        end
        
        pVars(nRun) = pVar;
        
        % plot
        if plotStatus
            clf(f);
            idx = randi(nUnits);
            subplot(2,4,1);
            hold on;
            imagesc(currTargets(iTarget,:)); colormap(jet), colorbar;
            axis tight; set(gca, 'clim', [0 1])
            title('real');
            set(gca,'Box','off','TickDir','out','FontSize',14);
            
            subplot(2,4,2);
            hold on;
            imagesc(R(iTarget,:)); colormap(jet), colorbar;
            axis tight; set(gca, 'clim', [0 1])
            title('model');
            set(gca,'Box','off','TickDir','out','FontSize',14);
            
            subplot(2,4,[3 4 7 8]);
            hold all;
            plot(tRNN,R(iTarget(idx),:), 'linewidth', 1.5);
            plot(tData,currTargets(iTarget(idx),:), 'linewidth', 1.5);
            axis tight; set(gca, 'ylim', [-0.1 1], 'Box','off','TickDir', 'out', 'FontSize', 14)
            ylabel('activity'); xlabel('time (s)'),
            legend('model', 'real', 'location', 'eastoutside')
            title(['run ', num2str(nRun)])
            
            subplot(2,4,5);
            hold on;
            plot(pVars(1:nRun)); ylabel('pVar');
            set(gca, 'ylim', [-0.1 1], 'Box', 'off', 'TickDir', 'out', 'FontSize', 14);
            title(['current pVar=', num2str(pVars(nRun), '%.3f')])
            
            subplot(2,4,6);
            hold on;
            plot(chi2(1:nRun)); ylabel('chi2');
            set(gca, 'ylim', [-0.1 1], 'Box','off','TickDir', 'out', 'FontSize', 14);
            title(['current chi2=', num2str(chi2(nRun), '%.3f')])
            drawnow;
        end
        
%         if nRun == 1
%             fprintf(num2str(nRun))
%         elseif mod(nRun, 100) == 0
%             fprintf('\n')
%             fprintf(num2str(nRun))
%         else
%             fprintf('.')
%         end
        
    end
    
    % save J for next link
    JTrls(:, :, iTrl) = J;
    
    % sidebar to save J separately in the hopes of making loading
    % RNNs faster (and crossreferencing
%     JTensor = [rnnSubDir, RNNname, '_JTrl.mat'];
%     
%     if exist(JTensor, 'file') && iTrl ~= 1
%         load(JTensor);
%     elseif ~exist(JTensor, 'file') && iTrl == 1
%         JAllTrl = JTrls;
%     else
%         disp('Jsaving mistake perhaps')
%         keyboard
%     end
    
%     JAllTrl(:, :, iTrl) = J;
%     if saveMdl
%         save(JTensor, 'JAllTrl', '-v7.3') % will cross reference with extracted from makeRNNPlots
%     end
%     clear JAllTrl
    
    % package up and save outputs at the end of training for each link
    RNN = struct;
    
    if setID(iTrl) == 0
        rnnParams = struct( ...
            'iTrl',                 iTrl, ...
            'doSmooth',             doSmooth, ...
            'smoothWidth',          smoothWidth, ...
            'doSoftNorm',           doSoftNorm, ...
            'normByRegion',         normByRegion, ...
            'rmvOutliers',          rmvOutliers, ...
            'outliers',             outliers, ...
            'dtFactor',             dtFactor, ...
            'g',                    g, ...
            'alpha',                alpha, ...
            'tauRNN',               tauRNN, ...
            'tauWN',                tauWN, ...
            'ampInWN',              ampInWN, ...
            'nRunTot',              nRunTot, ...
            'nRunTrain',            nRunTrain, ...
            'nRunFree',             nRunFree, ...
            'nonlinearity',         nonlinearity, ...
            'resetPoints',          resetPoints, ... % will need to change
            'nUnits',               nUnits, ...
            'nTrls',                nTrls, ...
            'nSets',                nSets, ...
            'setID',                setID(iTrl), ...
            'arrayUnit',            {spikeInfo.array}, ...
            'arrayRegions',         {arrayRgns});
    else
        rnnParams = [];
    end
    
    RNN.mdl = struct( ...
        'iTrl',                 iTrl, ...
        'prevTrl',              prevTrl, ...
        'setID',                setID(iTrl), ...
        'RMdlSample',           rModelSample, ...
        'tRNN',                 tRNN, ...
        'dtRNN',                dtRNN, ...
        'targets',              currTargets, ...
        'tData',                tData, ...
        'dtData',               dtData, ...
        'J',                    J, ...
        'J0',                   J0, ...
        'chi2',                 chi2, ...
        'pVars',                pVars, ...
        'stdData',              stdData(iTrl), ...
        'inputWN',              [], ...
        'iTarget',              iTarget, ...
        'iNonTarget',           iNonTarget, ...
        'params',               rnnParams );
    
    if saveMdl
        save([rnnSubDir, RNNname, '_set', num2str(setID(iTrl)), '_trial', num2str(iTrl), '.mat'],'RNN', '-v7.3')
    end
    
    clear RNN
    % toc
    
end

end






