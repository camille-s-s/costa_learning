function [Coder] = generateCoder(BHVout,BHVerror,NEURALevnt,NEURALtime)
% created by Vincent Costa
% updatedd by Hua Tang
% Dec 8,2018 @ NIMH

Coder.BHVout=BHVout;
Coder.BHVerror=BHVerror;%% what's BHVerror means? not complated?
Coder.validtrls = find(Coder.BHVerror==0);

% Coder.stimlock = %you need to extract this from the neural event times; it is the onset of the stimulus presentation in the timeframe in which the neural data was recorded.
% Coder.responselock = %you need to extract this from the neural event times; it is the onset of the stimulus presentation plus the choice reaction time in the timeframe in which the neural data was recorded.

for h = 1:size(Coder.validtrls,1)
    Coder.fixlock(h,1) = NEURALtime{Coder.validtrls(h)}(NEURALevnt{Coder.validtrls(h)}==3);
    Coder.stimlock(h,1) = NEURALtime{Coder.validtrls(h)}(NEURALevnt{Coder.validtrls(h)}==5);
    
    if sum(NEURALevnt{Coder.validtrls(h)}==8)>0 
        Coder.juicelock(h,1) = min(NEURALtime{Coder.validtrls(h)}(NEURALevnt{Coder.validtrls(h)}==8));
    elseif sum(NEURALevnt{Coder.validtrls(h)}==10)>0
        Coder.juicelock(h,1) = min(NEURALtime{Coder.validtrls(h)}(NEURALevnt{Coder.validtrls(h)}==10));
    end;
end;

Coder.choices = BHVout(:,37);
Coder.reward =  BHVout(:,38);
% Coder.rewardpt = cat(1,0,BHVout(2:end,38)); %p
Coder.rewardpt = cat(1,0,BHVout(1:end-1,38));
Coder.picid = BHVout(:,21:23);
Coder.trlssincenov = BHVout(:,25);

[Qsa,Qtran] = mdpChoice_appx_ratio_v3_PDPorSER([],[],BHVout(:,37),BHVout(:,38),BHVout(:,21:23),2,0.9);

Coder.Qsa = Qsa;
Coder.Qe = Qsa-Qtran;
Coder.Qt = Qtran;
Coder.Qb_mc = Qtran-repmat(mean(Qtran,2),1,3);

for c = 1:size(BHVout(:,37))
    Coder.cQsa(c,1) = Coder.Qsa(c,Coder.choices(c,1));
    Coder.cQe(c,1) = Coder.Qe(c,Coder.choices(c,1));
    Coder.cQt(c,1) = Coder.Qt(c,Coder.choices(c,1));
    Coder.cQb_mc(c,1) = Coder.Qb_mc(c,Coder.choices(c,1));
end;

nur = unique(BHVout(:,33));
cho_apriori = BHVout(:,33);
cho_aprioricat = BHVout(:,33);

for r = 1:max(size(nur))
    k = BHVout(:,33)==nur(r);
    if nur(r)<=0.4,
        cho_aprioricat(k,1)=1;
    elseif nur(r)>=0.6
        cho_aprioricat(k,1)=3;
    elseif nur(r)>=0.4 & nur(r)<=0.6;
        cho_aprioricat(k,1)=2;
    end;
end;

Coder.cApriori = cho_apriori;
Coder.cAprioriCat = cho_aprioricat;
Coder.direction = BHVout(:,32);
Coder.orientation = rem(Coder.direction,2)+1; % the old version should without "+1".
Coder.srt = BHVout(:,35);

%This assigns the stimulus identities to each of the chosen cues

stim = Coder.picid*0;
nstim = 1;
for s = 1:3,
    k = cat(1,1,find(BHVout(:,20+s)==1),size(BHVout,1));
    for h = 1:(max(size(k))-1)
        stim(k(h):k(h+1),s) = nstim;
        nstim = nstim+1;
    end;
    clear k;
end;

Coder.stim = stim; 
for c = 1:size(Coder.choices,1)
    Coder.chosenstim(c,1) = Coder.stim(c,Coder.choices(c,1));
end;

%This sorts the stimulus identities by the a priori reward probability and
%relabels them within the reward probability category so they are
%approiately nested for the ANOVA analyses

for r = 1:3
    k = Coder.cAprioriCat==r;
    uk = unique(Coder.chosenstim(k,1));
    nstim = 1;
    for g = 1:max(size(uk));
        Coder.chosenstimsort(Coder.chosenstim(:,1)==uk(g),1) = nstim;
        nstim = nstim+1;
    end;
    clear k;
    clear uk;
end;