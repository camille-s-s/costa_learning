

% plot original J and trained J
c_lim = 0.1*max(max(abs(J)))*[-1 1];
figure;
subplot(1,2,1); hold all;
imagesc(J0([in_AMY in_SC in_VS],[in_AMY in_SC in_VS]));
axis square;
axis tight;
set(gca,'Box','off','TickDir','out','FontSize',14,'YDir','normal');
colormap(brewermap(100,'*RdBu'));
set(gca,'CLim',c_lim);

subplot(1,2,2); hold all;
imagesc(J([in_AMY in_SC in_VS],[in_AMY in_SC in_VS]));
axis square;
axis tight;
set(gca,'Box','off','TickDir','out','FontSize',14,'YDir','reverse');
colormap(brewermap(100,'*RdBu'));
set(gca,'CLim',c_lim);

V = axis;
plot(length(in_AMY)*[1 1],V(3:4),'k--');
plot((length(in_AMY)+length(in_SC))*[1 1],V(3:4),'k--');
plot((length(in_AMY)+length(in_SC)+length(in_VS))*[1 1],V(3:4),'k--');
plot(V(1:2),length(in_AMY)*[1 1],'k--');
plot(V(1:2),(length(in_AMY)+length(in_SC))*[1 1],'k--');
plot(V(1:2),(length(in_AMY)+length(in_SC)+length(in_VS))*[1 1],'k--');


%% plot distributions for full J and all submatrices
n_bins = 56;

c = [0 0 0;0.5 0.5 0.5];

figure;
hold all;
d = J;
d = d./sqrt(size(d,2));
    val = max(max(abs(d)));
    J_bins = linspace(-val-0.1,val+0.1,n_bins);
[n,x] = hist(reshape(d,numel(d),1),J_bins);
n = n./max(n);
plot(x,n,'Color',c(1,:),'LineWidth',1.5);
plot(x,n,'.','Color',c(1,:),'MarkerSize',20);

d = J0;
d = d./sqrt(size(d,2));
    val = max(max(abs(d)));
    J_bins = linspace(-val-0.1,val+0.1,n_bins);
[n,x] = hist(reshape(d,numel(d),1),J_bins);
n = n./max(n);
plot(x,n,'Color',c(2,:),'LineWidth',1.5);
plot(x,n,'.','Color',c(2,:),'MarkerSize',20);

set(gca,'Box','off','TickDir','out','FontSize',14,'YScale','log');

xlabel('DI Weight');
title('Full J');


%% Now plot submatrices
count = 1;
J_bins = -0.2:.005:0.2;
y_lim = [1e-4 1];

figure('Position',[100 100 900 900]);

%%%%%% AMYGDALA
subplot(3,3,count); hold all; count = count + 1;
hold all;
d = J(in_AMY,in_AMY);
d = d./sqrt(size(d,2));
[n,x] = hist(reshape(d,numel(d),1),J_bins);
n = n./max(n);
plot(x,n,'Color',c(1,:),'LineWidth',1.5);
plot(x,n,'.','Color',c(1,:),'MarkerSize',20);

d = J0(in_AMY,in_AMY);
d = d./sqrt(size(d,2));
[n,x] = hist(reshape(d,numel(d),1),J_bins);
n = n./max(n);
plot(x,n,'Color',c(2,:),'LineWidth',1.5);
plot(x,n,'.','Color',c(2,:),'MarkerSize',20);
set(gca,'Box','off','TickDir','out','FontSize',14,'YScale','log');
xlabel('DI Weight');
title('J(Amy,Amy)');
set(gca,'XLim',J_bins([1 end]));
set(gca,'YLim',y_lim);
ylabel('AMYGDALA');

subplot(3,3,count); hold all; count = count + 1;
hold all;
d = J(in_AMY,in_SC);
d = d./sqrt(size(d,2));
[n,x] = hist(reshape(d,numel(d),1),J_bins);
n = n./max(n);
plot(x,n,'Color',c(1,:),'LineWidth',1.5);
plot(x,n,'.','Color',c(1,:),'MarkerSize',20);

d = J0(in_AMY,in_SC);
d = d./sqrt(size(d,2));
[n,x] = hist(reshape(d,numel(d),1),J_bins);
n = n./max(n);
plot(x,n,'Color',c(2,:),'LineWidth',1.5);
plot(x,n,'.','Color',c(2,:),'MarkerSize',20);
set(gca,'Box','off','TickDir','out','FontSize',14,'YScale','log');
xlabel('DI Weight');
title('J(Amy,Sc)');
set(gca,'XLim',J_bins([1 end]));
set(gca,'YLim',y_lim);

subplot(3,3,count); hold all; count = count + 1;
hold all;
d = J(in_AMY,in_VS);
d = d./sqrt(size(d,2));
[n,x] = hist(reshape(d,numel(d),1),J_bins);
n = n./max(n);
plot(x,n,'Color',c(1,:),'LineWidth',1.5);
plot(x,n,'.','Color',c(1,:),'MarkerSize',20);

d = J0(in_AMY,in_VS);
d = d./sqrt(size(d,2));
[n,x] = hist(reshape(d,numel(d),1),J_bins);
n = n./max(n);
plot(x,n,'Color',c(2,:),'LineWidth',1.5);
plot(x,n,'.','Color',c(2,:),'MarkerSize',20);
set(gca,'Box','off','TickDir','out','FontSize',14,'YScale','log');
xlabel('DI Weight');
title('J(Amy,Vs)');
set(gca,'XLim',J_bins([1 end]));
set(gca,'YLim',y_lim);

%%%%%% ACC
subplot(3,3,count); hold all; count = count + 1;
hold all;
d = J(in_SC,in_AMY);
d = d./sqrt(size(d,2));
[n,x] = hist(reshape(d,numel(d),1),J_bins);
n = n./max(n);
plot(x,n,'Color',c(1,:),'LineWidth',1.5);
plot(x,n,'.','Color',c(1,:),'MarkerSize',20);

d = J0(in_SC,in_AMY);
d = d./sqrt(size(d,2));
[n,x] = hist(reshape(d,numel(d),1),J_bins);
n = n./max(n);
plot(x,n,'Color',c(2,:),'LineWidth',1.5);
plot(x,n,'.','Color',c(2,:),'MarkerSize',20);
set(gca,'Box','off','TickDir','out','FontSize',14,'YScale','log');
xlabel('DI Weight');
title('J(Sc,Amy)');
set(gca,'XLim',J_bins([1 end]));
set(gca,'YLim',y_lim);
ylabel('ACC');

subplot(3,3,count); hold all; count = count + 1;
hold all;
d = J(in_SC,in_SC);
d = d./sqrt(size(d,2));
[n,x] = hist(reshape(d,numel(d),1),J_bins);
n = n./max(n);
plot(x,n,'Color',c(1,:),'LineWidth',1.5);
plot(x,n,'.','Color',c(1,:),'MarkerSize',20);

d = J0(in_SC,in_SC);
d = d./sqrt(size(d,2));
[n,x] = hist(reshape(d,numel(d),1),J_bins);
n = n./max(n);
plot(x,n,'Color',c(2,:),'LineWidth',1.5);
plot(x,n,'.','Color',c(2,:),'MarkerSize',20);
set(gca,'Box','off','TickDir','out','FontSize',14,'YScale','log');
xlabel('DI Weight');
title('J(SC,Sc)');
set(gca,'XLim',J_bins([1 end]));
set(gca,'YLim',y_lim);

subplot(3,3,count); hold all; count = count + 1;
hold all;
d = J(in_SC,in_VS);
d = d./sqrt(size(d,2));
[n,x] = hist(reshape(d,numel(d),1),J_bins);
n = n./max(n);
plot(x,n,'Color',c(1,:),'LineWidth',1.5);
plot(x,n,'.','Color',c(1,:),'MarkerSize',20);

d = J0(in_SC,in_VS);
d = d./sqrt(size(d,2));
[n,x] = hist(reshape(d,numel(d),1),J_bins);
n = n./max(n);
plot(x,n,'Color',c(2,:),'LineWidth',1.5);
plot(x,n,'.','Color',c(2,:),'MarkerSize',20);
set(gca,'Box','off','TickDir','out','FontSize',14,'YScale','log');
xlabel('DI Weight');
title('J(SC,Vs)');
set(gca,'XLim',J_bins([1 end]));
set(gca,'YLim',y_lim);


%%%%%% STRIATUM
subplot(3,3,count); hold all; count = count + 1;
hold all;
d = J(in_VS,in_AMY);
d = d./sqrt(size(d,2));
[n,x] = hist(reshape(d,numel(d),1),J_bins);
n = n./max(n);
plot(x,n,'Color',c(1,:),'LineWidth',1.5);
plot(x,n,'.','Color',c(1,:),'MarkerSize',20);

d = J0(in_VS,in_AMY);
d = d./sqrt(size(d,2));
[n,x] = hist(reshape(d,numel(d),1),J_bins);
n = n./max(n);
plot(x,n,'Color',c(2,:),'LineWidth',1.5);
plot(x,n,'.','Color',c(2,:),'MarkerSize',20);
set(gca,'Box','off','TickDir','out','FontSize',14,'YScale','log');
xlabel('DI Weight');
title('J(Vs,Amy)');
set(gca,'XLim',J_bins([1 end]));
set(gca,'YLim',y_lim);
ylabel('STRIATUM');

subplot(3,3,count); hold all; count = count + 1;
hold all;
d = J(in_VS,in_SC);
d = d./sqrt(size(d,2));
[n,x] = hist(reshape(d,numel(d),1),J_bins);
n = n./max(n);
plot(x,n,'Color',c(1,:),'LineWidth',1.5);
plot(x,n,'.','Color',c(1,:),'MarkerSize',20);

d = J0(in_VS,in_SC);
d = d./sqrt(size(d,2));
[n,x] = hist(reshape(d,numel(d),1),J_bins);
n = n./max(n);
plot(x,n,'Color',c(2,:),'LineWidth',1.5);
plot(x,n,'.','Color',c(2,:),'MarkerSize',20);
set(gca,'Box','off','TickDir','out','FontSize',14,'YScale','log');
xlabel('DI Weight');
title('J(Vs,Sc)');
set(gca,'XLim',J_bins([1 end]));
set(gca,'YLim',y_lim);

subplot(3,3,count); hold all; count = count + 1;
hold all;
d = J(in_VS,in_VS);
d = d./sqrt(size(d,2));
[n,x] = hist(reshape(d,numel(d),1),J_bins);
n = n./max(n);
plot(x,n,'Color',c(1,:),'LineWidth',1.5);
plot(x,n,'.','Color',c(1,:),'MarkerSize',20);

d = J0(in_VS,in_VS);
d = d./sqrt(size(d,2));
[n,x] = hist(reshape(d,numel(d),1),J_bins);
n = n./max(n);
plot(x,n,'Color',c(2,:),'LineWidth',1.5);
plot(x,n,'.','Color',c(2,:),'MarkerSize',20);
set(gca,'Box','off','TickDir','out','FontSize',14,'YScale','log');
xlabel('DI Weight');
title('J(Vs,Vs)');
set(gca,'XLim',J_bins([1 end]));
set(gca,'YLim',y_lim);



%% quick plot of pvar and chi2
figure;
subplot(2,1,1);
plot(pVars);
axis tight;
set(gca,'Box','off','TickDir','out','FontSize',14);

subplot(2,1,2);
plot(chi2);
axis tight;
set(gca,'Box','off','TickDir','out','FontSize',14);

