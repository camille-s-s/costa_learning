function [spikeCount]=novelty_spike(aTS,Event,Bin,varargin)
% novelty_spike.m - calculate spikeCount and SDF
% Hua Tang,Postdoc of Dr. Averbeck Lab @NIMH.
% Last update: Nov 21, 2021
%__________________________________________________________________________
%
% INPUT
% aTS:          timestemps of all spikes: neurons x 1 Cellarry with elements arrange as spikes x 1 matrix.
% centre:       event of time 0
% Bin:          parameters of bins
%
% varargin:
% SDF:          caculate spike density function as well.
%
% OUTPUT
% spikeCount:   spike count in each bin
% % spikeCount: spike density function
%__________________________________________________________________________

bin_cen=Bin.cen;
period=Bin.period;
width=Bin.width;

% centre: event of time 0
if strcmp(Bin.center,'cue')
    timelock = Event.cue; % cue, no need to *1000 here
elseif strcmp(Bin.center,'choice')
    timelock =Event.choice;
elseif  strcmp(Bin.center,'reward')
    timelock = Event.reward; % transfer ms to s.
end

for icell=1:length(aTS)
    if length(aTS)==1 % compatible for single unit
        tTS= aTS.timestamps;
    else
        tTS= aTS{icell,1}.timestamps;
    end
    for itrial=1:length(timelock)
        try
            indTS=find((tTS>timelock(itrial)+period(1)-width/2) & (tTS<=timelock(itrial)+period(2)+width/2));
            temTS=tTS(indTS)-timelock(itrial);
            
            tspike=[];
            for m=1:length(bin_cen)
                tspike(m)=numel(find((temTS>bin_cen(m)-width/2) & (temTS<=bin_cen(m)+width/2)));
            end
            
            spikeCount(icell,itrial,:)=tspike;
        catch
            spikeCount(icell,itrial,:)= nan(1,length(bin_cen));
        end
    end
end
end