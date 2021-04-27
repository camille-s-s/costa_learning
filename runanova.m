function runanova(filename,Coder)
% used to do n-way anova test (factor: ) for Noval Seeking Task.

fmo = matfile(filename,'Writable',true); % matfile:Access and change variables directly in MAT-files, without loading into memory?
icell = fmo.icell;

%dospkden = 1;
if ~isfield(icell,'spkden_200_10')
    
    ispkden = nan(size(Coder.choices,1),601);
    parfor h = 1:size(Coder.choices,1) % parallel for loop
        resplock = ((Coder.stimlock(h)*1000)+Coder.srt(h)); % time of response (saccade)
        ispkden(h,:) = genspkdenitrl([-3000 3000],(icell.timestamps(:,2))-resplock,10,200);
        %disp(h);
    end;
    
    icell.spkden_200_10 = ispkden; % bin:200 ms; step:10 ms;
else
    ispkden = icell.spkden_200_10;
end;

disp('Starting ANOVA');

% ANOVA factors
ievcat = Coder.cAprioriCat;
stimsort = Coder.chosenstimsort;
dir = Coder.direction;
ori = Coder.orientation;
ievresid = Coder.cQe-Coder.cApriori;
fev = Coder.cQt(:,1);
fevz = (fev-min(fev))/range(fev);
bon = Coder.cQb_mc(:,1);
bonz = (bon-0)/range(bon);
rew = Coder.reward;
rewpt = Coder.rewardpt;
tsn = Coder.trlssincenov;

Y = ispkden;
Y = sqrt(Y);
Y = Y-nanmean(reshape(Y,numel(Y),1))./(nanstd(reshape(Y,numel(Y),1))); % Y=Y-SEM

Xmat = {ievcat stimsort dir ori ievresid fevz bonz rew rewpt tsn+1};

vcon = [5 6 7 10]; % continuous factors

nestvar = zeros(max(size(Xmat)),max(size(Xmat)));
nestvar(2,1) = 1; % ??
nestvar(3,4) = 1; % ??

Xmodel = eye(max(size(Xmat)));

nbins = size(Y,2);
ypvalu = nan(size(Xmodel,1),nbins); 
yresid = nan(size(bon,1),nbins); % ??
anova_tbl = nan(size(Xmodel,1)+2,6,nbins);
statssv = cell(nbins,1); % save ANOVA stat forms?

parfor t = 1:nbins
    [ypvalu(:,t),tbl,statssv{t}] = anovan(Y(:,t),Xmat,'nested',nestvar,'model',Xmodel,'continuous',vcon,'display','off');
    %statssv{t} = stats;
    anova_tbl(:,:,t) = cat(2,cell2mat(tbl(2:end,2:4)),cat(1,cell2mat(tbl(2:end,5)),-999),cat(1,cell2mat(tbl(2:end,6:7)),[-999 -999; -999 -999]));
    %ypvalu(:,t) = p;
    yresid(:,t) = statssv{t}.resid;
end;

icell.resp_200_10_ypvalu = ypvalu;
icell.resp_200_10_anova_tbl = anova_tbl;
icell.resp_200_10_yresid = yresid;
icell.resp_200_10_statssv = statssv;

fmo.icell = icell;
fmo.Coder = Coder;

% % % close all;
% % % subplot(1,2,1);
% % % plot([-3000:10:3000],-log10(ypvalu([1 5 6 7 10],:)'),'LineWidth',2);
% % % legend({'ievcat' 'ievres' 'fev' 'bon' 'tsn'});
% % % ylimc = get(gca,'YLim');
% % % set(gca,'YLim',[0 max([6 max(ylimc)])]);
% % % subplot(1,2,2);
% % % plot([-3000:10:3000],-log10(ypvalu([2 3 4 8 9],:)'),'LineWidth',2);
% % % legend({'stim' 'dir' 'ori' 'rew' 'rept'});
% % % ylimc = get(gca,'YLim');
% % % set(gca,'YLim',[0 max([6 max(ylimc)])]);
% % % set(gcf,'Position',[226 543 1107 412]);
