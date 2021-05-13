clearvars
close all
load('/Users/camille/Dropbox (BrAINY Crew)/csRNN/data/from_matt/rnn_fitD_perc100_run1.mat')
o3_AddCurrentsToTD_runs;

monkey = 'D';
[~,td] = getTDidx(master_td,'monkey',monkey);


bad_idx = mean(getSig(td,'AMY_spikes'),1) < 0.05;
for trial = 1:length(td)
    td(trial).AMY_spikes = td(trial).AMY_spikes(:,~bad_idx);
    td(trial).Curr_AmyAmy = td(trial).Curr_AmyAmy(:,~bad_idx);
    td(trial).Curr_VsAmy = td(trial).Curr_VsAmy(:,~bad_idx);
    td(trial).Curr_ScAmy = td(trial).Curr_ScAmy(:,~bad_idx);
end
bad_idx = mean(getSig(td,'SC_spikes'),1) < 0.05;
for trial = 1:length(td)
    td(trial).SC_spikes = td(trial).SC_spikes(:,~bad_idx);
    td(trial).Curr_AmySc = td(trial).Curr_AmySc(:,~bad_idx);
    td(trial).Curr_VsSc = td(trial).Curr_VsSc(:,~bad_idx);
    td(trial).Curr_ScSc = td(trial).Curr_ScSc(:,~bad_idx);
end
bad_idx = mean(getSig(td,'VS_spikes'),1) < 0.05;
for trial = 1:length(td)
    td(trial).VS_spikes = td(trial).VS_spikes(:,~bad_idx);
    td(trial).Curr_AmyVs = td(trial).Curr_AmyVs(:,~bad_idx);
    td(trial).Curr_VsVs = td(trial).Curr_VsVs(:,~bad_idx);
    td(trial).Curr_ScVs = td(trial).Curr_ScVs(:,~bad_idx);
end

which_type = '';
switch which_type
    case '_inh'
        c_lim = 1*[-1 1];
    case '_exc'
        c_lim = 1*[-1 1];
    otherwise
        c_lim = 4*[-1 1];
end

softFac = 0.05;

% get the sort from the full spikes in juice
trial = 2;
a = td(trial).AMY_spikes;
a = smooth_data(a,1,100);
[~,idx] = max(a,[],1);
[~,idx_A] = sort(idx);
a = td(trial).SC_spikes;
a = smooth_data(a,1,100);
[~,idx] = max(a,[],1);
[~,idx_S] = sort(idx);
a = td(trial).VS_spikes;
a = smooth_data(a,1,100);
[~,idx] = max(a,[],1);
[~,idx_V] = sort(idx);

trial = 2;
figure('Position',[100 100 1600 1200]);
count = 1;
%%%% FIRST PLOT AMYGDALA
% full spikes
subplot(3,5,count); hold all; count = count + 1;
a = td(trial).AMY_spikes;
a = a./repmat(mean(abs(a),1),size(a,1),1);
% a = a./repmat(max(abs(a),[],1),size(a,1),1);
imagesc(a(:,idx_A)');
axis tight;
set(gca,'Box','off','TickDir','out','FontSize',14);
ylabel('Amygdala');
title({'Full Amydala','Activity'});
set(gca,'CLim',[-5 5])
% AMY currents
subplot(3,5,count); hold all; count = count + 1;
a = td(trial).(['Curr_AmyAmy' which_type]);
a = a./repmat(mean(abs(a),1),size(a,1),1);
imagesc(a(:,idx_A)');
axis tight;
set(gca,'Box','off','TickDir','out','FontSize',14);
ylabel('Amygdala');
title({'Currents:','Amygdala to Amygdala'});
set(gca,'CLim',c_lim);
% SC currents
subplot(3,5,count); hold all; count = count + 1;
a = td(trial).(['Curr_ScAmy' which_type]);
a = a./repmat(mean(abs(a),1),size(a,1),1);
imagesc(a(:,idx_A)');
axis tight;
set(gca,'Box','off','TickDir','out','FontSize',14);
ylabel('Amygdala');
title({'Currents:','Subcollosal ACC to Amygdala'});
set(gca,'CLim',c_lim);
% VS currents
subplot(3,5,count); hold all; count = count + 1;
a = td(trial).(['Curr_VsAmy' which_type]);
a = a./repmat(mean(abs(a),1),size(a,1),1);
imagesc(a(:,idx_A)');
axis tight;
set(gca,'Box','off','TickDir','out','FontSize',14);
ylabel('Amygdala');
title({'Currents:','Striatum to Amygdala'});
set(gca,'CLim',c_lim);
% Sum currents
subplot(3,5,count); hold all; count = count + 1;
a = td(trial).(['Curr_AmyAmy' which_type]) + ...
    td(trial).(['Curr_ScAmy' which_type]) + ...
    td(trial).(['Curr_VsAmy' which_type]);
a = a./(repmat(mean(abs(a),1),size(a,1),1)+softFac);
imagesc(a(:,idx_A)');
axis tight;
set(gca,'Box','off','TickDir','out','FontSize',14);
ylabel('Amygdala');
title({'Currents:','Sum of All Sources'});
set(gca,'CLim',[0 2])


%%%% NEXT PLOT ACC
% full spikes
subplot(3,5,count); hold all; count = count + 1;
a = td(trial).SC_spikes;
a = a./repmat(mean(abs(a),1),size(a,1),1);
imagesc(a(:,idx_S)');
axis tight;
set(gca,'Box','off','TickDir','out','FontSize',14);
ylabel('Subcollosal ACC');
title({'Full Subcollosal ACC','Activity'});
set(gca,'CLim',[-5 5])
% AMY currents
subplot(3,5,count); hold all; count = count + 1;
a = td(trial).(['Curr_AmySc' which_type]);
a = a./repmat(mean(abs(a),1),size(a,1),1);
imagesc(a(:,idx_S)');
axis tight;
set(gca,'Box','off','TickDir','out','FontSize',14);
ylabel('Subcollosal ACC');
title({'Currents:','Amygdala to Subcollosal ACC'});
set(gca,'CLim',c_lim);
% SC currents
subplot(3,5,count); hold all; count = count + 1;
a = td(trial).(['Curr_ScSc' which_type]);
a = a./repmat(mean(abs(a),1),size(a,1),1);
imagesc(a(:,idx_S)');
axis tight;
set(gca,'Box','off','TickDir','out','FontSize',14);
ylabel('Subcollosal ACC');
title({'Currents:','Subcollosal ACC to Subcollosal ACC'});
set(gca,'CLim',c_lim);
% VS currents
subplot(3,5,count); hold all; count = count + 1;
a = td(trial).(['Curr_VsSc' which_type]);
a = a./repmat(mean(abs(a),1),size(a,1),1);
imagesc(a(:,idx_S)');
axis tight;
set(gca,'Box','off','TickDir','out','FontSize',14);
ylabel('Subcollosal ACC');
title({'Currents:','Striatum to Subcollosal ACC'});
set(gca,'CLim',c_lim);
% Sum currents
subplot(3,5,count); hold all; count = count + 1;
a = td(trial).(['Curr_AmySc' which_type]) + ...
    td(trial).(['Curr_ScSc' which_type]) + ...
    td(trial).(['Curr_VsSc' which_type]);
a = a./(repmat(mean(abs(a),1),size(a,1),1)+softFac);
imagesc(a(:,idx_S)');
axis tight;
set(gca,'Box','off','TickDir','out','FontSize',14);
ylabel('Subcollosal ACC');
title({'Currents:','Sum of All Sources'});
set(gca,'CLim',c_lim);
set(gca,'CLim',[0 2])


%%%% NEXT PLOT STRIATUM
% full spikes
subplot(3,5,count); hold all; count = count + 1;
a = td(trial).VS_spikes;
a = a./repmat(mean(abs(a),1),size(a,1),1);
imagesc(a(:,idx_V)');
axis tight;
set(gca,'Box','off','TickDir','out','FontSize',14);
ylabel('Striatum');
title({'Full Striatum','Activity'});
set(gca,'CLim',[-5 5])
% AMY currents
subplot(3,5,count); hold all; count = count + 1;
a = td(trial).(['Curr_AmyVs' which_type]);
a = a./repmat(mean(abs(a),1),size(a,1),1);
imagesc(a(:,idx_V)');
axis tight;
set(gca,'Box','off','TickDir','out','FontSize',14);
ylabel('Striatum');
title({'Currents:','Amygdala to Striatum'});
set(gca,'CLim',c_lim);
% SC currents
subplot(3,5,count); hold all; count = count + 1;
a = td(trial).(['Curr_ScVs' which_type]);
a = a./repmat(mean(abs(a),1),size(a,1),1);
imagesc(a(:,idx_V)');
axis tight;
set(gca,'Box','off','TickDir','out','FontSize',14);
ylabel('Striatum');
title({'Currents:','Subcollosal ACC to Striatum'});
set(gca,'CLim',c_lim);
% VS currents
subplot(3,5,count); hold all; count = count + 1;
a = td(trial).(['Curr_VsVs' which_type]);
a = a./repmat(mean(abs(a),1),size(a,1),1);
imagesc(a(:,idx_V)');
axis tight;
set(gca,'Box','off','TickDir','out','FontSize',14);
ylabel('Striatum');
title({'Currents:','Striatum to Striatum'});
set(gca,'CLim',c_lim);
% Sum currents
subplot(3,5,count); hold all; count = count + 1;
a = td(trial).(['Curr_AmyVs' which_type]) + ...
    td(trial).(['Curr_ScVs' which_type]) + ...
    td(trial).(['Curr_VsVs' which_type]);
a = a./(repmat(mean(abs(a),1),size(a,1),1)+softFac);
imagesc(a(:,idx_V)');
axis tight;
set(gca,'Box','off','TickDir','out','FontSize',14);
ylabel('Striatum');
title({'Currents:','Sum of All Sources'});
set(gca,'CLim',[0 2])



colormap(brewermap(100,'*RdGy'));







%%
monkey = 'D';
[~,td] = getTDidx(master_td,'monkey',monkey);

bad_idx = mean(getSig(td,'AMY_spikes'),1) < 0.1;
for trial = 1:length(td)
    td(trial).AMY_spikes = td(trial).AMY_spikes(:,~bad_idx);
end
bad_idx = mean(getSig(td,'SC_spikes'),1) < 0.1;
for trial = 1:length(td)
    td(trial).SC_spikes = td(trial).SC_spikes(:,~bad_idx);
end
bad_idx = mean(getSig(td,'VS_spikes'),1) < 0.1;
for trial = 1:length(td)
    td(trial).VS_spikes = td(trial).VS_spikes(:,~bad_idx);
end

trial = 2;
a = td(trial).AMY_spikes;
a = smooth_data(a,1,100);
[~,idx] = max(a,[],1);
[~,idx_A] = sort(idx);
% normFacA = max(abs(getSig(td,'AMY_spikes')),[],1);
normFacA = mean(abs(getSig(td,'AMY_spikes')),1);

a = td(trial).SC_spikes;
a = smooth_data(a,1,100);
[~,idx] = max(a,[],1);
[~,idx_S] = sort(idx);
% normFacS = max(abs(getSig(td,'SC_spikes')),[],1);
normFacS = mean(abs(getSig(td,'SC_spikes')),1);

a = td(trial).VS_spikes;
a = smooth_data(a,1,100);
[~,idx] = max(a,[],1);
[~,idx_V] = sort(idx);
% normFacV = max(abs(getSig(td,'VS_spikes')),[],1);
normFacV = mean(abs(getSig(td,'VS_spikes')),1);

softFac = 0.05;

cm = brewermap(100,'*RdGy');

figure('Position',[100 100 500 1200]);
count = 1;
for trial = 1:length(td)
    
    subplot(length(td),3,count); hold all; count = count + 1;
    a = td(trial).AMY_spikes;
    a = a./(repmat(normFacA,size(a,1),1)+softFac);
    imagesc(a(:,idx_A)');
    axis tight;
    set(gca,'Box','off','TickDir','out','FontSize',14);
    set(gca,'CLim',[0 2]);
    if trial == 1
        title('Amygdala');
    end
    
    switch td(trial).condition
        case 1
            ylabel('Control');
        case 3
            ylabel('Juice');
        case 4
            ylabel('Water');
        case 5
            ylabel('No reward');
    end
    
    
    subplot(length(td),3,count); hold all; count = count + 1;
    a = td(trial).SC_spikes;
    a = a./(repmat(normFacS,size(a,1),1)+softFac);
    imagesc(a(:,idx_S)');
    axis tight;
    set(gca,'Box','off','TickDir','out','FontSize',14);
    set(gca,'CLim',[0 2]);
    if trial == 1
        title('Subcollossal ACC');
    end
    
    subplot(length(td),3,count); hold all; count = count + 1;
    a = td(trial).VS_spikes;
    a = a./(repmat(normFacV,size(a,1),1)+softFac);
    imagesc(a(:,idx_V)');
    axis tight;
    set(gca,'Box','off','TickDir','out','FontSize',14);
    set(gca,'CLim',[0 2]);
    if trial == 1
        title('Striatum');
    end
    
end

colormap(cm);




%% plot currents
close all
cm = brewermap(100,'*RdBu');

td = td(2:4);
c = [-3.5 3.5];

a = td(2).Curr_AmyAmy;
a = a./repmat(mean(abs(a),1),size(a,1),1);
[~,idx] = max(a,[],1);
[~,idx] = sort(idx);

figure;
count = 1;
for trial = 1:length(td)
    subplot(length(td),3,count); hold all; count = count + 1;
    a = td(trial).Curr_AmyAmy;
    a = a./repmat(mean(abs(a),1),size(a,1),1);
    imagesc(a(:,idx)');
    axis tight;
    set(gca,'Box','off','TickDir','out','FontSize',14);
    set(gca,'CLim',c);
    title('Amygdala to Amygdala');
    
    switch td(trial).condition
        case 1
            ylabel('Control');
        case 3
            ylabel('Juice');
        case 4
            ylabel('Water');
        case 5
            ylabel('No reward');
    end
    
    
    subplot(length(td),3,count); hold all; count = count + 1;
    a = td(trial).Curr_ScAmy;
    a = a./repmat(mean(abs(a),1),size(a,1),1);
    imagesc(a(:,idx)');
    axis tight;
    set(gca,'Box','off','TickDir','out','FontSize',14);
    set(gca,'CLim',c);
    title('Subcollossal ACC to Amygdala');
    
    subplot(length(td),3,count); hold all; count = count + 1;
    a = td(trial).Curr_VsAmy;
    a = a./repmat(mean(abs(a),1),size(a,1),1);
    imagesc(a(:,idx)');
    axis tight;
    set(gca,'Box','off','TickDir','out','FontSize',14);
    set(gca,'CLim',c);
    title('Striatum to Amygdala');
    
end

colormap(cm);


