function plot_costa_RNN_progress(f, nUnits, targets, R, tRNN, tData, nRun, pVars, MSE, trainRNN, fTitle)

fTitle(strfind(fTitle, '_')) = ' ';
set(0, 'currentfigure', f);
nSP_Data = size(targets, 2);
clf(f);
idx = randi(nUnits);
gridPos = {1, 2, [3 4 7 8], 5, 6};

if trainRNN
    axsTitles = {'targets', 'model', ['training run ', num2str(nRun)], ['training pVar=', num2str(pVars(nRun), '%.3f')], ['sum MSE=', num2str(MSE(nRun), '%.3e')]};
    axsXLbls = {'time (s)', 'mdl timestep', 'time (s)', 'run #', 'run #'};
    axsYLims = [0.5 nUnits + 0.5; 0.5 nUnits + 0.5; -0.1 0.4; -0.1 1.1; -0.1 1];

else
    axsTitles = {'targets', 'model', ['test trl'], ['pVar=', num2str(pVars(nRun), '%.3f')], ['mean/sum MSE=', num2str(mean(MSE), '%.3e'), '/', num2str(sum(MSE), '%.2f')]};
    axsXLbls = {'time (s)', 'mdl timestep', 'time (s)', 'run #', 'timestep'};
    axsYLims = [0.5 nUnits + 0.5; 0.5 nUnits + 0.5; -0.1 0.4; -0.1 1.1; 0 1.1 * round(max(MSE), 2, 'significant')];

end
    axsYLbls = {'units', 'units', 'activity', 'pVar', 'MSE'};

axs = arrayfun( @(i) subplot(2, 4, gridPos{i}, 'NextPlot', 'add', 'box', 'off', 'tickdir', 'out', 'fontsize', 12, 'fontweight', 'bold'), 1 : length(gridPos) );


% format axes
arrayfun( @(i) set(axs(i), 'clim', [0 1]), 1 : 2)
arrayfun( @(i) colormap(axs(i), 'jet'), 1 : 2)
arrayfun( @(i) colorbar(axs(i)), 1 : 2)
arrayfun( @(i) axis(axs(i), 'tight'), 1 : 3)
arrayfun( @(i) grid(axs(i), 'minor'), [4 5])
arrayfun( @(i) set(axs(i), 'ylim', axsYLims(i, :)), 1 : 5)

% label axes
arrayfun( @(i) title(axs(i), axsTitles{i}, 'fontsize', 14), 1 : 5)
arrayfun( @(i) xlabel(axs(i), axsXLbls{i}), 1 : 5)
arrayfun( @(i) ylabel(axs(i), axsYLbls{i}), 1 : 5)

% imagesc model and targets
imagesc(axs(1), targets),
imagesc(axs(2), R),

% figure title
text(axs(2), 0.75 * range(get(axs(2), 'xlim')), 1.15 * max(get(axs(2), 'ylim')), fTitle, 'fontweight', 'bold', 'fontsize', 18)

% example trace model vs target comparison of a single unit
plot(axs(3), tRNN, R(idx, :), 'linewidth', 1.5)
plot(axs(3), tData(2 : end), targets(idx, :), 'linewidth', 1.5)
legend(axs(3), 'model', 'target', 'location', 'northeast')

if trainRNN % chi2 over training runs
    plot(axs(4), pVars(1 : nRun)) % only get a curve for training runs since nRunTrain > 1
    plot(axs(5), MSE(1 : nRun), 'k', 'linewidth', 1.5);
else % chi2 over each timestep of one test run through a trial
    plot(axs(5), MSE(1 : nSP_Data), 'k', 'linewidth', 1.5);
    set(axs(5), 'xlim', [1 nSP_Data]);
    line(axs(5), [1 nSP_Data], [mean(MSE(1 : nSP_Data)) mean(MSE(1 : nSP_Data))], 'linestyle', ':', 'linewidth', 1, 'color', [0.2 0.2 0.2])
end

drawnow;

end