% used to create spikecounts for single Units.
% for utah arrays.
% Written by Hua Tang,Postdoc of Dr. Averbeck Lab, @NIMH in Dec,2021.
% last update: Dec 22, 2021

close all; clear;clc;
warning('off')

pathin='D:\PFC_8ARRAY\novelty\sorted files\';
pathout= 'D:\PFC_8ARRAY\novelty\sorted files\spikecounts\';

cd (pathin)
[file,path,indx] = uigetfile('*.mat', 'multiselect','on');
if isequal(file,0)
    error 'No file was selected'
    return
elseif ischar(file)
    list = {file};
elseif iscell(file)
    list=file';
end


for il=1:length(list)
    anip{il,1}=list{il,1}(1:9);
end

use=unique(anip);

%%% setting
Bin.period = [-2 2]; % raw neural file only +- 1000 ms;
Bin.width = 0.2; % bin width
Bin.step = 0.05; 
Bin.cen=Bin.period(1):Bin.step:Bin.period(2);
Bin.center='cue';

cd (pathout)

for ise=1:length(use)
    tind=find(strcmp(anip,use(ise)))';
    nCoders=[];spikeCount=[];tt=0;
    for isee=tind
        tt=tt+1;
        load([pathin,list{isee}])
        fName=list{isee}(1:9)
        nCoder=novelty_coder_transfer(Coder);
        Neuron{tt}=Neurons;
        [spikeCount{tt}]=novelty_spike(iCells,nCoder,Bin);
    end
    
    spikeCount=cat(1,spikeCount{1},spikeCount{2});
    Neurons=cat(1,Neuron{1},Neuron{2});
    
    save([fName, '_SC_', Bin.center, '_', num2str(Bin.width*1000), '_', num2str(Bin.step*1000)],'spikeCount','Neurons','Bin','nCoder')
end