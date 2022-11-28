
function plot_costa_RNN_param_comparisons(f2, axs, rnnName, nUnits, targets, R, tRNN, tData, nRun, pVars, MSE, axNum)

set(0, 'currentfigure', f2);
axsXLbls = {'time (s)'};
axsYLims = [-0.1 0.4];
axsYLbls = {'activity'};
idx = randi(nUnits);

% do this outside of function
% f2 = figure('color', 'w', 'Position', [100 100 1900 750]);
% axs = arrayfun( @(i) subplot(1, 4, i, 'NextPlot', 'add', 'box', 'off', 'tickdir', 'out', 'fontsize', 12, 'fontweight', 'bold'), 1 : 4);
% paramText = sprintf(['g = \t', num2str(g), '\nP_0 = \t', num2str(alpha), '\ntau_WN = \t', num2str(tauWN), '\nw_WN = \t', num2str(ampInWN), '\nn_iter = \t', num2str(nRunTrain), '\ndt_Data = \t', num2str(dtData), '\ndt_RNN = \t', num2str(dtRNN)  ]); % this at the end!


% axis title
if axNum == 1
    axsTitle = {[rnnName, ' run ', num2str(nRun)], ['pVar = ', num2str(pVars(nRun), '%.2f')], ['mean/sum MSE = ', num2str(mean(MSE), '%.2e'), ' / ', num2str(sum(MSE), '%.2f')]};
else
    axsTitle = {['run ', num2str(nRun)], ['pVar = ', num2str(pVars(nRun), '%.2f')], ['mean/sum MSE = ', num2str(mean(MSE), '%.2e'), ' / ', num2str(sum(MSE), '%.2f')]};
end

title(axs(axNum), axsTitle, 'fontsize', 14)

% format axes
axis(axs(axNum), 'tight')
set(axs(axNum), 'ylim', axsYLims)

% arrayfun( @(i) axis(axs(i), 'tight'), 1 : 3)
% arrayfun( @(i) set(axs(i), 'ylim', axsYLims(i, :)), 1 : 3)

% label axes

xlabel(axs(axNum), axsXLbls)
ylabel(axs(axNum), axsYLbls)

% arrayfun( @(i) xlabel(axs(i), axsXLbls{i}), 1 : 3)
%arrayfun( @(i) ylabel(axs(i), axsYLbls{i}), 1 : 3)


% example trace model vs target comparison of a single unit
plot(axs(axNum), tRNN, R(idx, :), 'linewidth', 1.5)
plot(axs(axNum), tData(2 : end), targets(idx, :), 'linewidth', 1.5)

if axNum == 1
    legend(axs(axNum), 'model', 'target', 'location', 'northwest')
end


end