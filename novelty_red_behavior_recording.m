function [BHVout,BHV] = novelty_red_behavior_recording(BHVfile,mname,noem,old,vpwd)
%%%%%%%%%%%%%%%%%%%%%%%%%%%% BHVout parameters %%%%%%%%%%%%%%%%%%%%%%%%%%%%
% uodated by Hua Tang, Feb 20, 2019
% 1: mname % monkey name (the order in the name pool)
% 2-7: absolute first Trial Start Time
% 8-13: absolute current Trial Start Time
% 14: un-break trial number 
% 15-17: rrprob_aval (reward probability)
% 18-20: timeinarray (the times a specific pic continue appeared)
% 21-23: picid (when & where the novel image appeared)
% 24:arraydex£¨??£©
% 25:trlsincenovel (trial number since novel image)
% 26:dropin_loc (novel trial' location )
% 27: dropinrewprob (novel trial' reward probability )
% 28-30: sort(rrprob_aval,2)
% 31: chosenovel (whether chosen the novel image)
% 32: cloc (Choice location based on scenario file)
% 33: crewprob ()
% 34: csvl (Choice saccade velocity)
% 35: csrt (Choice saccadic reaction time)
% 36: choice_em (which image has been chosen)
% 37: choice_eyelog (choice)
% 38: juice (reward)
% 39: nVisRepofChoice (how many times the chosen image has been shown)
% 40: stimos_t-stimem_t 
% 41: 22222222 
% 42: fixholdtime (fixation holding time)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if nargin < 3, noem = 0; end;
if nargin < 4, old = 0; end;
if nargin < 5, vpwd = pwd; end;
dosaccades = 1;

monk = {'pei'; 'corbu'; 'chumpy'; 'quibble'; 'bandit'; 'eto'; 'golgi'; 'faraday'; 'moog'; 'nyquist'; 'orazio'; 'hartmut'; 'formaggi'; 'alfredo'; 'gremolata'; 'bean'; 'ender'; 'voltaire'; 'waldo'};
mname = find(strcmp(mname,monk)==1);

try
    BHV = bhv_read([vpwd,'/',BHVfile]);
catch
    disp('Using alternate bhv_read_trunc .....');
    BHV = bhv_read_trunc([vpwd,'/',BHVfile]);
end;

close all;

saclocs = [-2.3126; -0.8290; 0; 0.8290; 2.3126; 3.1416];
% set(gca,'XTick',[1 2 3 4 5 6],'XTickLabel',{'LU'; 'UP'; 'RU'; 'LD'; 'DW'; 'RD'})


% try
%     BHV = bhv_read([pwd,'//',BHVfile]);
% catch
%     disp('Using alternate bhv_read_trunc .....');
%     BHV = bhv_read_trunc([pwd,'//',BHVfile]);
% end;

nfev    = 7500;
niter   = 5;
%niter = 10;

blkincthresh = find(cat(1,0,diff(BHV.ConditionNumber))<0)-1;

if isempty(blkincthresh)==1,
    blkincthresh(1,1) = size(BHV.ConditionNumber,1);
else
    disp('error: trials in opposite order!!!') %% hua
    blkincthresh(1,1) = size(BHV.ConditionNumber,1); %% trials in opposite order
end;

z = BHV.TrialNumber;
z = z(1:blkincthresh);
z = z(BHV.TrialError(1:blkincthresh)==0);% | BHV.TrialError(1:blkincthresh)==5);
%z = z(BHV.TrialError(blkincthresh:end)==0);% | BHV.TrialError(1:blkincthresh)==5);
%z = z(1:length(z)-1);
disp(size(z));
pause(1);

picid = zeros(max(size(z)),1);

dl = 0;
dic = 0;
dropin_loc = [];
dropin_cond = [];
fc_other = [];
sm = [1 1 2];
stim_mix = [];
timeinarray = zeros(max(size(z),3));
picid = zeros(max(size(z)),3);
Lfig = figure();
centrls = zeros(max(size(z)),1);

picrewstimcoder = zeros(BHV.Stimuli.NumPics,2);
trlsfromNov = 0;
treps = [0 0 0];
bt = [];
for n = 1:max(size(z))
    
    if BHV.InfoByCond{BHV.ConditionNumber(z(n))}.diloc>0
        trlsfromNov = 0;
    end;
    
    trlsincenovel(n,1) = trlsfromNov;
    
    trlsfromNov = trlsfromNov+1;
    
    % Juice Record based on BHV.RewardRecord field
    tjuice = (sum(BHV.RewardRecord(z(n)).RewardOnTime) > 0);
    % Juice Record based on Event Markers coded in BHV.Condition Numbers
    ejuice = sum(BHV.CodeNumbers{z(n)}==8);
    % Check to make sure that juice is doubly and accurately recorded in
    % the BHV.RewardRecord and BHV.CodeNumbers fields for the current trial
    chkjuice(n,1) = ((tjuice+0)==(ejuice+0));
    % Log Juice Reward
    juice(n,1) = tjuice;
    
    % Time(in ms due to use of 1000 Hz sampling rate) at which the Stimulus Array was Presented
    so(n,1) = BHV.ObjectStatusRecord(z(n)).Time(2);
    if old==1,
        soff(n,1) = so(n,1)+1000;
    else
        soff(n,1) = BHV.ObjectStatusRecord(z(n)).Time(4);
    end;
    % X and Y coordinates in degrees of visual angle
    X = BHV.AnalogData{1,z(n)}.EyeSignal(:,1);
    Y = BHV.AnalogData{1,z(n)}.EyeSignal(:,2);
    
    % Register Choice Behavior Based on Saccades and Obtain Reaction Times
    if dosaccades==1,
        
        % Sub-routine to obtain saccade parameters
        [S,~] = getsaccades(X,Y,BHV.PixelsPerDegree,0);
        
        % Reaction time pulled from event markers for trials in which the
        % monkey did not sufficiently hold the stimulus following his choice
        % This is vestigal code when only correct trials are considered.
        %         if sum(BHV.CodeNumbers{z(n)}==17)==1,
        %             srt(n,1)=min(BHV.CodeTimes{z(n)}(BHV.CodeNumbers{z(n)}==6))-so(n,1);
        %         else
        
        % For the current trial compute the saccadic reaction times
        % relative to the onset of the stimulus array
        %csrt = S.saccadeStartIdx - so(n,1);
        
        % Compute the distance from center fixation for the peak of each saccade
        try
            S.saccadedist = sqrt((S.saccadeXdir.^2)+(S.saccadeYdir.^2));
        catch
            bt(n) = n;
            disp(['Not A Saccade: ',num2str(n)]);
            plot(X,Y); title(num2str(n));
            pause(3);
            disp('');
            continue;
        end;
        
        S.saccadeloc = atan2(S.saccadeXdir,S.saccadeYdir);
        for m = 1:length(S.saccadeloc),
            if S.saccadedist(m)<=4.2426,
                S.saccadeloccat(m) = -999;
            else
                saclactmp = (saclocs - S.saccadeloc(m)).^2;
                S.saccadeloccat(m) = min(find(saclactmp==min(saclactmp)));
            end;
        end;
        poststimhold = min(find(((S.saccadeStartIdx>(so(n,1)+50) & S.saccadedist>4.2426) & S.saccadeStartIdx<soff(n,1))==1));
        % Choice saccadic reaction time
        if isempty(poststimhold),
            disp('Empty'); disp(n);
            poststimhold = min(find(((S.saccadeStartIdx>(so(n,1)+50) & S.saccadedist>4.2426))));
            if isempty(poststimhold),
                disp('Problem'); disp(n);
                bt(n) = n;
                continue;
            end;
        end;
        csrt(n,1) = S.saccadeStartIdx(poststimhold)-so(n,1);
        % Choice saccade velocity
        csvl(n,1) = S.saccadeVelocityIdx(poststimhold);
        % Choice number of saccades
        cnsc(n,1) = sum((S.saccadeStartIdx>(so(n,1)+50) & S.saccadedist>4.2426) & S.saccadeStartIdx<soff(n,1));
        % Choice location based on scenario file
        cloc(n,1) = S.saccadeloccat(poststimhold);
        
        for ll = 1:3,
            tmpto = regexp(BHV.TaskObject{BHV.ConditionNumber(z(n)),(ll+1)},',','split');
            locxy(ll,1) = str2num(tmpto{2});
            locxy(ll,2) = str2num(strtok(tmpto{3},')'));
            distfromchoice(ll,1) = sqrt( ((S.saccadeXdir(poststimhold)-locxy(ll,1)).^2 + (S.saccadeYdir(poststimhold)-locxy(ll,2)).^2) );
            
        end;
        
        arraydex(n,1) = locxy(find(locxy(:,1)==0),2);
        
        choice_eyelog(n,1) = find( distfromchoice(:,1)==min(distfromchoice(:,1)) );        
    end;    
    clear S; clear X; clear Y; clear locxy;   
    fco = 0;
       
    if n > 1,        
        for k = 1:3,            
            pre_pic = regexp(BHV.TaskObject{BHV.ConditionNumber(z(n-1)),k+1},',','split');
            cur_pic = regexp(BHV.TaskObject{BHV.ConditionNumber(z(n)),k+1},',','split');
            
            picmatch = strcmp(pre_pic{1},cur_pic{1});
            % IF
            % the current and previous picture don't match
            % AND
            % string describing the current picture contains 'nov'
            % (i.e. the array returned by the call to strfind is not empty)
            % picid(n,k) is assigned a value of 1
            % ! NB ! strfind is used instead of strcmpin for backward
            % compatiblity we prior stimulus naming conventions
            if picmatch==0 && isempty(strfind(cur_pic{1},'nov'))==0,
                picid(n,k) = picmatch+1;
                dic = 1;
                dl = k;
                fco = 1;
                sm(1,k) = 2;
                treps(k) = 0;
                
                % IF
                % the current and previous picture don't match
                % AND
                % string describing the current picture contains 'fam'
                % (i.e. the array returned by the call to strfind is not empty)
                % picid(n,k) is assigned a value of 1
            elseif picmatch==0 && isempty(strfind(cur_pic{1},'fam'))==0,
                picid(n,k) = picmatch+1;
                dic = 1;
                dl = k;
                fco = 1;
                sm(1,k) = 1;
                treps(k) = 0;                
            end;
        end;
    end;
    
    timeinarray(n,:) = treps;
    treps = treps+1;
    
    dropin_loc = cat(1,dropin_loc,dl);
    dropin_cond = cat(1,dropin_cond,dic);
    fc_other = cat(1,fc_other,fco);
    stim_mix = cat(1,stim_mix,sm);
    
    if noem == 1,
        choice_em(n,1) = choice_eyelog(n,1)+500;
        %        choice_loc(n,1) = 999; %atan2(locofchoices(choice_em(n,1)-501,1),locofchoices(choice_em(n,1)-501,2));
    else
        choice_em(n,1) = min(BHV.CodeNumbers{z(n)}(BHV.CodeNumbers{z(n)}>500));
        %        choice_loc(n,1) = atan2(locofchoices(choice_em(n,1)-501,1),locofchoices(choice_em(n,1)-501,2));
    end;
        
    block_cond(n,1:2) = [BHV.BlockIndex(z(n)) BHV.ConditionNumber(z(n)) ];
    if old == 0,        
        rrprob_aval(n,1:3) = [  BHV.InfoByCond{BHV.ConditionNumber(z(n)),1}.rew1...
            BHV.InfoByCond{BHV.ConditionNumber(z(n)),1}.rew2...
            BHV.InfoByCond{BHV.ConditionNumber(z(n)),1}.rew3];
        %for rpa_n = 1:3,
        %    if rrprob_aval(n,rpa_n)>.85, rrprob_aval(n,rpa_n)=.75; elseif rrprob_aval(n,rpa_n)<.15, rrprob_aval(n,rpa_n)=.25; end;
        %end;
    else
        rpa = (BHV.CodeNumbers{z(n)}(BHV.CodeNumbers{z(n)}>=400 & BHV.CodeNumbers{z(n)}<=500)-400)/10;
        %for rpa_n = 1:3,
        %    if rpa(rpa_n)>.85, rpa(rpa_n)=.75; elseif rpa(rpa_n)<.15, rpa(rpa_n)=.25; end;
        %end;
        rrprob_aval(n,1:3) = [  rpa(1,1) rpa(2,1) rpa(3,1) ]; clear rpa;
    end;
    
end;

cp = [1 2 3];
n = 1:max(size(z))';
bt = unique(bt);
if ~isempty(bt)    
    z = z(~ismember(n,bt),:);
    choice_em = choice_em(~ismember(n,bt),:);
    choice_eyelog = choice_eyelog(~ismember(n,bt),:);
    rrprob_aval = rrprob_aval(~ismember(n,bt),1:3);
    timeinarray = timeinarray(~ismember(n,bt),:);
    trlsincenovel = trlsincenovel(~ismember(n,bt),1);
    juice = juice(~ismember(n,bt),:);
    picid = picid(~ismember(n,bt),:);
    csrt = csrt(~ismember(n,bt),1);
    csvl = csvl(~ismember(n,bt),1);
    cnsc = cnsc(~ismember(n,bt),1);
    cloc = cloc(~ismember(n,bt),1);
    arraydex = arraydex(~ismember(n,bt),1); 
end;

%Check if multiple probs > 3
if max(size(unique(rrprob_aval)))>3
    rrprob_aval = reshape(rrprob_aval,size(rrprob_aval,1)*size(rrprob_aval,2),1);
    for k = .20:.05:.95
        if ~isempty(find(rrprob_aval==k))
            if k<=min(rrprob_aval)
                frp = .2;
            elseif k< (max(rrprob_aval)-.20)
                frp = .5;
            elseif k>= max(rrprob_aval)
                frp = .8;
            end;
            rrprob_aval(find(rrprob_aval==k),1)=frp;
        end;
    end;
    disp('N.B.  Edited rrprob_aval');
    rrprob_aval = reshape(rrprob_aval,size(picid,1),size(picid,2));
end;

choice_em = (choice_em - min(choice_em))+1;
if dosaccades==1,
    ccagree=sum(choice_em==choice_eyelog)/size(choice_em,1);
    disp(['% Agreement in Choice Assignment: ',num2str(ccagree)]);
end;
%

choice = choice_em;

for n = 1:max(size(z)),
    for k = 1:3,
        cur_pic = regexp(BHV.TaskObject{BHV.ConditionNumber(z(n)),k+1},',','split');
        %disp(cur_pic{1});
        if choice_em(n,1)==k,
            for hh = 1:BHV.Stimuli.NumPics,
                det(hh) = max([strfind(cur_pic{1},BHV.Stimuli.PIC(1,hh).Name) 0]);
            end;
            pdex = find(det>0);
            picrewstimcoder(pdex,1:2) = [picrewstimcoder(pdex,1)+juice(n) picrewstimcoder(pdex,2)+1];
            break;
        end;
    end;
end;


for n = 1:size(rrprob_aval,1)
    crewprob(n,1) = rrprob_aval(n,choice(n,1));
end;


% Fraction of Choices for each Reward Probability for N trials out from
% Introduction of Novel Stimulus
rp = unique(rrprob_aval)';
choice_cnt_pick = zeros(size(choice,1),max(size(rp)));
choice_cnt_nopick = zeros(size(choice,1),max(size(rp)));
for trial = 1 : size(choice,1)
    for kk = 1:3,
        % Identify the column index for the current reward probability
        % If two choices have the sample probability they will be indexed
        % into the same column
        cc = find(rp == rrprob_aval(trial,kk));
        % Log the number of times the stimulus of reward probabily X has
        % been presented up to this point
        cr = timeinarray(trial,kk)+1;
        % Increment the number of times this specific reward probability
        % has been picked at this point since introducing a novel stimulus
        choice_cnt_nopick(cr,cc) = choice_cnt_nopick(cr,cc)+1;
        % Do the same if the choice corresponds to the current stimulus
        if choice(trial)==kk,
            choice_cnt_pick(cr,cc) = choice_cnt_pick(cr,cc)+1;
        end;
    end;
end;

% Archival Code for recycling pictures the monkey has seen but that has a
% chance reward probability
%useasfam = find((picrewstimcoder(:,1)./picrewstimcoder(:,2)) < .6 & (picrewstimcoder(:,1)./picrewstimcoder(:,2)) > .4);
%for i = 1:size(useasfam,1)
%        disp(BHV.Stimuli.PIC(1,useasfam(i)).Name);
%end;

for h = 1:size(z,1)
    if dropin_loc(h,1)==0,
        dropinrewprob(h,1)=999;
    else
        dropinrewprob(h,1) = rrprob_aval(h,dropin_loc(h,1));
    end;
end;

chosenovel = (choice_em==dropin_loc)+0;

for m = 1:size(timeinarray,1)
    nVisRepofChoice(m,1) = timeinarray(m,choice_em(m,1));
end;

BHVout = cat(2,repmat([mname BHV.AbsoluteTrialStartTime(1,:)],size(z,1),1),BHV.AbsoluteTrialStartTime(z,:),z,rrprob_aval,timeinarray,picid,arraydex,trlsincenovel,dropin_loc,dropinrewprob,sort(rrprob_aval,2),chosenovel,cloc,crewprob,csvl,csrt,choice_em,choice_eyelog,juice,nVisRepofChoice);

for r = 1:max(numel(rp)),
    for t = 1:1:15,
        cnp(t,r) = mean(chosenovel(min(timeinarray,[],2)==(t-1) & dropinrewprob==rp(r),1));
    end;
end;

for m = 1:size(BHVout,1)
    
    k = (BHV.CodeNumbers{BHVout(m,8+numel(BHV.AbsoluteTrialStartTime(1,:)))}==4);
    stimem_t = min(BHV.CodeTimes{BHVout(m,8+numel(BHV.AbsoluteTrialStartTime(1,:)))}(k));
    stimos_t = BHV.ObjectStatusRecord(BHVout(m,8+numel(BHV.AbsoluteTrialStartTime(1,:)))).Time(2);
    
    k = (BHV.CodeNumbers{BHVout(m,8+numel(BHV.AbsoluteTrialStartTime(1,:)))}==3);
    fixholdtime = min(BHV.CodeTimes{BHVout(m,8+numel(BHV.AbsoluteTrialStartTime(1,:)))}(k));
    
    em_os_diff(m,1:3) = [stimos_t-stimem_t 22222222 fixholdtime];
end;

BHVout(:,end-4) = BHVout(:,end-4) - em_os_diff(m,1);

BHVout(:,(34:36)++numel(BHV.AbsoluteTrialStartTime(1,:))) = em_os_diff;

plot(cnp); pause(1);
legend('low','medium','high')

%svf = [BHVfile,'_red_behav.mat'];
%save(svf,'BHV','BHVout');
