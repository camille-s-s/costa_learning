% used to merge behavioral files with spike files from MKsort.
% Written by Ramon Bartolo,Post Doc in Dr. Bruno Averbeck Lab.
% Updated by Hua Tang;
% NIMH,NIH
% Dec 8, 2018.

close all; clear;clc;
DataDir = 'F:\NIH-Research\PFC_ARRAY_DATA';%'D:\NIH-Research\PFC_ARRAY_DATA';
monkey = 'W';
AcDate = '20160205';

% lode coder files
DataDir2=[DataDir,'\',monkey,AcDate,'\'];
load([DataDir2,'novelty_', monkey, '_', AcDate, '_coder.mat'])

% load unit files
for NIP=2
    excel=['channel_list.xlsx'];
    sheet=[monkey, AcDate,'NIP',num2str(NIP)];
    [Unum Utxt] = xlsread([DataDir, '\' ,excel],sheet);
    
    %% combining
    DataDir3=[DataDir2,sheet,'\']
    if ~exist([DataDir3,'neuron'],'dir')
        mkdir([DataDir3,'neuron']);
    end
    
    cd ([DataDir3,'sorting\'])
    for i=1:length(Unum)
        fff='000';
        temp_Unum=num2str(Unum(i));
        UnitName=[Utxt{i},'_',fff(1:3-numel(temp_Unum)),temp_Unum];
        
        load(UnitName)
        try
            %             if waveforms.sorted ==1
            if sum(waveforms.units)>500
                [C, ~, ic]=unique(waveforms.units); % sorted units index, start from unsorted unit
                if sum(C)>0
                    ind=find(C>0);
                    for m=ind(1):ind(end)
                        tInd=find(ic==C(m)); % sorted units index
                        icell.spike=waveforms.waves(:,tInd)';
                        icell.timestamps(:,1)=C(m)*ones(length(tInd),1);
                        icell.timestamps(:,2)=waveforms.spikeTimes(:,tInd)';
                        icell.channel=Unum(i);
                        icell.date=AcDate;
                        
                        save([DataDir3,'neuron','\','icells_',sheet,'_',temp_Unum,'_',num2str(C(m))],'icell','Coder')
                        clear icell
                    end
                else
                    disp(['Unsorted channel: ', UnitName])
                end
                
            end
        catch
            warning(['Error: ', UnitName])
        end
    end
end
disp ('well done!!')