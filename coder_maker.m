clear;close all;clc


cd 'f:\NIH-Research\PFC_ARRAY_DATA\w20160211\w20160211NIP1'
load('novelty-waldo-02-11-2016-BHVeventsNIP1.mat');


 cd 'F:\NIH-Research\PFC_ARRAY_DATA\w20160211'
[BHVout] = novelty_red_behavior_recording('novelty-waldo-02-11-2016.bhv','waldo',0,0);


cd 'F:\NIH-Research\PFC_ARRAY_DATA\model'
Coder = generateCoder(BHVout,BHVerror,NEURALevnt,NEURALtime)

cd 'f:\NIH-Research\PFC_ARRAY_DATA\w20160211'
save('novelty_w_20160211_coder','Coder')



% % figure
% % plot(Coder.Qsa(:,1),'r')
% % 
% % 
% % AA=Coder.BHVout==Coder2.BHVout;
% % numel(find(AA==0))