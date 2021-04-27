clear; close all;

DataDir = 'F:\NIH-Research\PFC_ARRAY_DATA';
Date = '20160211';
monkey = 'w'; %'W';

DataDir2 = fullfile(DataDir, [monkey, Date]);

%%%% Find Behavioral file name and get Session Name
t = dir(fullfile(DataDir2,'*.bhv'));
if ~isempty(t)
    [~,BHVname,~] = fileparts(t.name);  
else
    error('BHV file not found...')
end

NIP = 2;
timingthres = 2; %ms

%%%% Find Neural file name
DataDir3=[DataDir2,'\',monkey,Date,'NIP',num2str(NIP)];  

t = dir(fullfile(DataDir3,'*.nev'));
if ~isempty(t)
    if numel(t)==1
        [~,RIPPLEname,~] = fileparts(t.name);
    else
        [~,RIPPLEname,~] = fileparts(t(NIP).name);
    end
else
    error('NEUROSHARE (RIPPLE) file not found...')
end
clear t

%%%% LOAD MONKEYLOGIC (BHV) DATA %%%%
disp('Reading BHV file...')
disp(fullfile(DataDir2,BHVname))
tic
BHV = bhv_read(fullfile(DataDir2,BHVname));
toc

%% need to combine these two scripts together soomthly!
% cd (DataDir2)
% [BHVout] = novelty_red_behavior_recording('novelty-waldo-02-05-2016.bhv','waldo',0,0);

trofst = [4,200]; %%??
if strcmp(BHV.ExperimentName,'novelty'); 
    trofst = [5,1000]; 
end
behavrtrials = length(BHV.CodeNumbers);
%bhvtrialnum(:,1) = cellfun(@(x) x(4)-200, BHV.CodeNumbers);
%bhvevntcode(:,1) = vertcat(BHV.CodeNumbers{:});
bhvevntcode = []; bhvevnttime = []; bhvtrialnum = [];
for i = 1:length(BHV.CodeNumbers)
    bhvtrialnum(i,1) = BHV.CodeNumbers{i}(trofst(1))-trofst(2);
    cn(:,1) = BHV.CodeNumbers{i}; cn(:,2) = bhvtrialnum(i,1);
    ct(:,1) = BHV.CodeTimes{i}  ; ct(:,2) = bhvtrialnum(i,1);
    bhvevntcode = vertcat(bhvevntcode,cn);
    bhvevnttime = vertcat(bhvevnttime,ct);
    clear cn ct
end

[choiceinfo, otherinfo] = getTrialInfo(BHV);

BHVexpnm = BHV.ExperimentName;
BHVevent = BHV.CodeNumbers;
BHVtimes = BHV.CodeTimes;
BHVerror = BHV.TrialError;
BHVeye   = BHV.AnalogData;
BHVinfo  = [choiceinfo, otherinfo];


tic
%  %%%OPEN THE RIPPLE (NEUROSHARE) FILE
%  disp('Opening Ripple File...')
%  ff = fullfile(RippleDir,tt{df});
%  [ns_status, hFile] = ns_OpenFile(fullfile(DataDir,RIPPLEname));

%%% GET & SORT EVENT MARKERS FROM RIPPLE FILES
disp('Reading Event Codes from NEV file...')
ff = fullfile(DataDir3,RIPPLEname);
[fileok, EventTimes, EventCode] = ExtractRippleData(ff,'Event',0);
toc

%%% VALIDATING EVENT- CODES/TIMES
FTS = false;
while ~FTS
    num9 = sum(EventCode(1:3)==9);
    if num9 < 3
        EventTimes(1,:) = [];
        EventCode(1,:) = [];
    else
        FTS = true;
    end
end

%get neural trial numbers
thistr = 0;
for i = 3:length(EventCode)
    if sum(EventCode(i-2:i)==9)==3
        thistr = thistr+1;
        neutrialnum(thistr,1) = EventCode(i+1)-200;
    end
end
if numel(bhvtrialnum)>numel(neutrialnum)
    firstBHVtrial = bhvtrialnum(1);
    firstNEUtrial = neutrialnum(1);
    lastBHVtrial = bhvtrialnum(end);
    lastNEUtrial = neutrialnum(end);
    if firstBHVtrial > firstNEUtrial
        PreRecTrials = find(bhvtrialnum<firstNEUtrial);
        bhvtrialnum(PreRecTrials) = [];
        BHVinfo(PreRecTrials,:) = [];
        BHVevent(PreRecTrials) = [];
        BHVtimes(PreRecTrials) = [];
        BHVerror(PreRecTrials) = [];
        BHVeye  (PreRecTrials) = [];
        clear PreRecTrials
    end
    if lastBHVtrial > lastNEUtrial
        PostRecTrials = find(bhvtrialnum>lastNEUtrial);
        bhvtrialnum(PostRecTrials) = [];
        BHVinfo(PostRecTrials,:) = [];
        BHVevent(PostRecTrials) = [];
        BHVtimes(PostRecTrials) = [];
        BHVerror(PostRecTrials) = [];
        BHVeye  (PostRecTrials) = [];
        clear PostRecTrials
    end
    clear firstBHVtrial firstNEUtrial
    clear lastBHVtrial lastNEUtrial
end

trials = bhvtrialnum;
WrongEvntCode = [];
TimeStampSBS(trials) = cell({[]});
TimingErrorTable = [];
timingerrors = 0;
for ntr = 1:length(trials)
    TimeStampSBS{trials(ntr)}(:,1) = BHVtimes{trials(ntr)}-min(BHVtimes{trials(ntr)});
    TimeStampSBS{trials(ntr)}(:,2) = (EventTimes(bhvevntcode(:,2)==trials(ntr))-min(EventTimes(bhvevntcode(:,2)==trials(ntr))))*1000;
    TimeStampSBS{trials(ntr)}(:,3) = -(diff(TimeStampSBS{trials(ntr)},[],2));
    TimeStampSBS{trials(ntr)}(2:end,4:5) = diff(TimeStampSBS{trials(ntr)}(:,1:2),[],1);
    TimeStampSBS{trials(ntr)}(:,6) = round(TimeStampSBS{trials(ntr)}(:,5));
    TimeStampSBS{trials(ntr)}(:,7) = abs(diff(TimeStampSBS{trials(ntr)}(:,[4,6]),[],2));
    temtab = TimeStampSBS{trials(ntr)}; temtab(:,8) = trials(ntr);
    TimingErrorTable = vertcat(TimingErrorTable, temtab); clear temtab;
    largediff = abs(TimeStampSBS{trials(ntr)}(:,3))>timingthres;
    timingerror(trials(ntr),1) = sum(largediff);
    if timingerror(trials(ntr))>0
        timingerrors = timingerrors+timingerror(trials(ntr));
        WrongEvntCode(ntr,1:timingerror(trials(ntr))) = find(largediff);
        disp([num2str(timingerror(trials(ntr))) ' large time mismatch (>' num2str(timingthres) 'ms) in trial ' num2str(ntr)])
        TimeStampSBS{trials(ntr)}(largediff,:)
    end
end

for i = 1:length(EventTimes)-2
    TrialStart(i) = sum(EventCode(i:i+2)==9)==3;
end

neuraltrials = sum(double(TrialStart));
trial_start = find(TrialStart);
for i = 1:neuraltrials
    if i<neuraltrials
        NEURALevnt{i}(:,1) = EventCode (trial_start(i):trial_start(i+1)-1);
        NEURALtime{i}(:,1) = EventTimes(trial_start(i):trial_start(i+1)-1);
    else
        NEURALevnt{i}(:,1) = EventCode (trial_start(i):end);
        NEURALtime{i}(:,1) = EventTimes(trial_start(i):end);
    end
    nrltrialnum(i,1) = NEURALevnt{i}(4)-200;
end
clear EventTimes EventCode

% save(fullfile(SaveDir,[BHVname '-BHVeventsNIP' num2str(NIP) '.mat']),'BHVexpnm','BHVevent','BHVinfo','BHVtimes','BHVerror','BHVeye','NEURALevnt','NEURALtime')
save(fullfile(DataDir3,[BHVname '-BHVeventsNIP' num2str(NIP) '.mat']),'BHVevent','BHVtimes','BHVinfo','BHVerror','BHVeye','NEURALevnt','NEURALtime','TimingErrorTable','WrongEvntCode')