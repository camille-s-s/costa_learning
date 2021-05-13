%% plot fishtwo style current state space for each brain region
monkeys = {'D'}; %,'H'};

figure('Position',[100 100 1200 800]);

for iMonk = 1:length(monkeys)
    monkey = monkeys{iMonk};
    
    load('/Users/camille/Dropbox (BrAINY Crew)/csRNN/data/from_matt/rnn_fitD_perc100_run1.mat')
    o3_AddCurrentsToTD_runs;
    
    [~,td] = getTDidx(master_td,'monkey',monkey);
    count = 1;

    
    which_type = ''; % '', '_inh', '_exc'
    which_regions = {'Amy','Sc','Vs'};
    
    c = [0.5 0.5 0.5; brewermap(2,'RdBu'); 0 0 0];
    
    switch monkey
        case 'D'
            if isempty(which_type)
                y_lim =2*[-1 1];
            else
                y_lim = 2.5 * [-1 1];
            end
        case 'H'
            if isempty(which_type)
                y_lim = 1.5*[-1 1];
            else
                y_lim = 2 * [-1 1];
            end
    end
    
    which_region = which_regions{1};
    
    switch which_region
        case 'Amy'
            region_name = 'Amygdala';
        case 'Sc'
            region_name = 'Subcollosal ACC';
        case 'Vs'
            region_name = 'Striatum';
    end
    
    
    ax = zeros(1,3);
    ax(1) = subplot(length(monkeys),3,count); hold all; count = count + 1;
    for trial = 1:length(td)
        plot(NaN,'LineWidth',3,'Color',c(trial,:));
    end
    b = getSig(td(1),{['Curr_Amy' which_region which_type '_pca'],1});b = b - repmat(b(1,:),size(b,1),1);
    for trial = 1:length(td)
        a = getSig(td(trial),{['Curr_Amy' which_region which_type '_pca'],1});
        %         a = a - repmat(a(1,:),size(a,1),1);
        %         a = a - b;
        plot(1:1500,a(:,1),'LineWidth',2,'Color',c(trial,:));
        plot(1,a(1,1),'.','MarkerSize',30,'Color',c(trial,:));
    end
    xlabel('Time (ms)');
    ylabel(['Monkey ' monkey]);
    axis square; axis tight;
    set(gca,'Box','off','TickDir','out','FontSize',14);
    title(['Amygdala to ' region_name ' Currents']);
    set(gca,'YLim',y_lim);
    
    
    ax(2) = subplot(length(monkeys),3,count); hold all; count = count + 1;
    for trial = 1:length(td)
        plot(NaN,'LineWidth',3,'Color',c(trial,:));
    end
    b = getSig(td(1),{['Curr_Sc' which_region which_type '_pca'],1});b = b - repmat(b(1,:),size(b,1),1);
    for trial = 1:length(td)
        a = getSig(td(trial),{['Curr_Sc' which_region which_type '_pca'],1});
        %         a = a - repmat(a(1,:),size(a,1),1);
        %         a = a - b;
        plot(1:1500,a(:,1),'LineWidth',2,'Color',c(trial,:));
        plot(1,a(1,1),'.','MarkerSize',30,'Color',c(trial,:));
    end
    xlabel('Time (ms)');
    axis square; axis tight;
    set(gca,'Box','off','TickDir','out','FontSize',14);
    title(['ACC to ' region_name ' Currents']);
    set(gca,'YLim',y_lim);
    
    
    ax(3) = subplot(length(monkeys),3,count); hold all; count = count + 1;
    for trial = 1:length(td)
        plot(NaN,'LineWidth',3,'Color',c(trial,:));
    end
    b = getSig(td(1),{['Curr_Vs' which_region which_type '_pca'],1});b = b - repmat(b(1,:),size(b,1),1);
    for trial = 1:length(td)
        a = getSig(td(trial),{['Curr_Vs' which_region which_type '_pca'],1});
        %         a = a - repmat(a(1,:),size(a,1),1);
        %         a = a - b;
        plot(1:1500,a(:,1),'LineWidth',2,'Color',c(trial,:));
        plot(1,a(1,1),'.','MarkerSize',30,'Color',c(trial,:));
    end
    xlabel('Time (ms)');
    axis square; axis tight;
    set(gca,'Box','off','TickDir','out','FontSize',14);
    title(['Striatum to ' region_name ' Currents']);
    set(gca,'YLim',y_lim);
    
end

legend({'Control','Juice','Water','NoReward'},'FontSize',14);




%% plot fishtwo style current state space for each brain region
monkeys = {'D','H'};

figure('Position',[100 100 1200 800]);

count = 1;
for iMonk = 1:length(monkeys)
    monkey = monkeys{iMonk};
    [~,td] = getTDidx(master_td,'monkey',monkey);
    
        
    which_type = ''; % '', '_inh', '_exc'
    which_regions = {'Amy','Sc','Vs'};
    which_region = which_regions{1};
    
    extra_name = '_pca'; % '' or '_pca'
    td = getNorm(td,{['Curr_Amy' which_region which_type extra_name],1:3});
    td = getNorm(td,{['Curr_Sc' which_region which_type extra_name],1:3});
    td = getNorm(td,{['Curr_Vs' which_region which_type extra_name],1:3});

    
    c = brewermap(3,'Dark2');
    
    switch monkey
        case 'D'
            if isempty(which_type)
%                 y_lim = 1.3*[-1 1];
y_lim = [-0.75 1];%[0.5 2];
            else
                y_lim = 2.5 * [-1 1];
            end
        case 'H'
            if isempty(which_type)
%                 y_lim = 1.3*[-1 1];
                y_lim = [-0.75 1];
            else
                y_lim = 2 * [-1 1];
            end
    end
    
    switch which_region
        case 'Amy'
            region_name = 'Amygdala';
        case 'Sc'
            region_name = 'Subcollosal ACC';
        case 'Vs'
            region_name = 'Striatum';
    end
    
    ax = zeros(1,4);
    for trial = [1 4 3 2]
        ax(1) = subplot(length(monkeys),length(td),count); hold all; count = count + 1;
        plot(NaN,'Color',c(1,:),'LineWidth',2);
        plot(NaN,'Color',c(2,:),'LineWidth',2);
        plot(NaN,'Color',c(3,:),'LineWidth',2);
        
        a = getSig(td(trial),{['Curr_Amy' which_region which_type extra_name '_norm'],1});
        aref = getSig(td(1),{['Curr_Amy' which_region which_type extra_name '_norm'],1});
%         a = a./mean(aref);
        a = a - repmat(a(1,:),size(a,1),1);
        plot(1:1500,a(:,1),'LineWidth',2,'Color',c(1,:));
        plot(1,a(1,1),'.','MarkerSize',30,'Color',c(1,:));
        
        a = getSig(td(trial),{['Curr_Sc' which_region which_type extra_name '_norm'],1});
        aref = getSig(td(1),{['Curr_Sc' which_region which_type extra_name '_norm'],1});
%         a = a./mean(aref);
        a = a - repmat(a(1,:),size(a,1),1);
        plot(1:1500,a(:,1),'LineWidth',2,'Color',c(2,:));
        plot(1,a(1,1),'.','MarkerSize',30,'Color',c(2,:));
        
        a = getSig(td(trial),{['Curr_Vs' which_region which_type extra_name '_norm'],1});
        aref = getSig(td(1),{['Curr_Vs' which_region which_type extra_name '_norm'],1});
%         a = a./mean(aref);
        a = a - repmat(a(1,:),size(a,1),1);
        plot(1:1500,a(:,1),'LineWidth',2,'Color',c(3,:));
        plot(1,a(1,1),'.','MarkerSize',30,'Color',c(3,:));
        
        xlabel('Time (ms)');
        if trial == 1
            ylabel(['Monkey ' monkey]);
        end
        axis square; axis tight;
        set(gca,'Box','off','TickDir','out','FontSize',14);
        set(gca,'YLim',y_lim);
        
        switch trial
            case 1
                title('Control');
            case 2
                title('Juice');
            case 3
                title('Water');
            case 4
                title('No Reward');
        end
    end
    
end
legend({'Amygdala','ACC','Striatum'});







%% plot fishtwo style current state space for each brain region
monkeys = {'D','H'};

figure('Position',[100 100 1200 800]);

count = 1;
for iMonk = 1:length(monkeys)
    monkey = monkeys{iMonk};
    [~,td] = getTDidx(master_td,'monkey',monkey);
    
        
    which_type = ''; % '', '_inh', '_exc'
    which_regions = {'Amy','Sc','Vs'};
    which_region = which_regions{1};
    
    extra_name = '_pca'; % '' or '_pca'
    td = getNorm(td,{['Curr_Amy' which_region which_type extra_name],1:3});
    td = getNorm(td,{['Curr_Sc' which_region which_type extra_name],1:3});
    td = getNorm(td,{['Curr_Vs' which_region which_type extra_name],1:3});

    
    c = brewermap(3,'Dark2');
    
    switch monkey
        case 'D'
            if isempty(which_type)
%                 y_lim = 1.3*[-1 1];
y_lim = [-0.75 1];%[0.5 2];
            else
                y_lim = 2.5 * [-1 1];
            end
        case 'H'
            if isempty(which_type)
%                 y_lim = 1.3*[-1 1];
                y_lim = [-0.75 1];
            else
                y_lim = 2 * [-1 1];
            end
    end
    
    switch which_region
        case 'Amy'
            region_name = 'Amygdala';
        case 'Sc'
            region_name = 'Subcollosal ACC';
        case 'Vs'
            region_name = 'Striatum';
    end
    
    ax = zeros(1,4);
    for trial = [1 4 3 2]
        ax(1) = subplot(length(monkeys),length(td),count); hold all; count = count + 1;
        plot(NaN,'Color',c(1,:),'LineWidth',2);
        plot(NaN,'Color',c(2,:),'LineWidth',2);
        plot(NaN,'Color',c(3,:),'LineWidth',2);
        
        a = getSig(td(trial),{['Curr_Amy' which_region which_type extra_name '_norm'],1});
        plot(1:1500,a(:,1),'LineWidth',2,'Color',c(1,:));
        plot(1,a(1,1),'.','MarkerSize',30,'Color',c(1,:));
        
        a = getSig(td(trial),{['Curr_Sc' which_region which_type extra_name '_norm'],1});
        plot(1:1500,a(:,1),'LineWidth',2,'Color',c(2,:));
        plot(1,a(1,1),'.','MarkerSize',30,'Color',c(2,:));
        
        a = getSig(td(trial),{['Curr_Vs' which_region which_type extra_name '_norm'],1});
        plot(1:1500,a(:,1),'LineWidth',2,'Color',c(3,:));
        plot(1,a(1,1),'.','MarkerSize',30,'Color',c(3,:));
        
        xlabel('Time (ms)');
        if trial == 1
            ylabel(['Monkey ' monkey]);
        end
        axis square; axis tight;
        set(gca,'Box','off','TickDir','out','FontSize',14);
        set(gca,'YLim',y_lim);
        
        switch trial
            case 1
                title('Control');
            case 2
                title('Juice');
            case 3
                title('Water');
            case 4
                title('No Reward');
        end
    end
    
end
legend({'Amygdala','ACC','Striatum'});