function plotCostaRNNProgress(f, nUnits, targets, R, tRNN, tData, nRun, pVars, chi2, trainRNN)

set(0, 'currentfigure', f);
nSP_Data = size(targets, 2);

idx = randi(nUnits);
gridPos = {1, 2, [3 4 7 8], 5, 6};

if trainRNN
    axsTitles = {'targets', 'model', ['training run ', num2str(nRun)], ['current training pVar=', num2str(pVars(nRun), '%.3f')], ['current training chi2=', num2str(chi2(nRun), '%.3f')]};
    axsXLbls = {'time (s)', 'mdl timestep', 'time (s)', 'run #', 'run #'};
else
    axsTitles = {'targets', 'model', ['testing run ', num2str(nRun)], ['final test pVar=', num2str(pVars(nRun), '%.3f')], ['final test chi2=', num2str(chi2(nSP_Data), '%.3f')]};
    axsXLbls = {'time (s)', 'mdl timestep', 'time (s)', 'run #', 'timestep'};
end
    axsYLbls = {'units', 'units', 'activity', 'pVar', 'chi2'};

axs = arrayfun( @(i) subplot(2, 4, gridPos{i}, 'NextPlot', 'add', 'box', 'off', 'tickdir', 'out', 'fontsize', 12, 'fontweight', 'bold'), 1 : length(gridPos) );

% format axes
arrayfun( @(i) set(axs(i), 'clim', [0 1]), 1 : 2)
arrayfun( @(i) colormap(axs(i), 'jet'), 1 : 2)
arrayfun( @(i) colorbar(axs(i)), 1 : 2)
arrayfun( @(i) axis(axs(i), 'tight'), 1 : 3)
arrayfun( @(i) set(axs(i), 'ylim', [-0.1 0.5]), [3 5])

% label axes
arrayfun( @(i) title(axs(i), axsTitles{i}), 1 : 5)
arrayfun( @(i) xlabel(axs(i), axsXLbls{i}), 1 : 5)
arrayfun( @(i) ylabel(axs(i), axsYLbls{i}), 1 : 5)

% imagesc model and targets
imagesc(axs(1), targets),
imagesc(axs(2), R),

% example trace model vs target comparison of a single unit
plot(axs(3), tRNN, R(idx, :), 'linewidth', 1.5)
plot(axs(3), tData(2 : end), targets(idx, :), 'linewidth', 1.5)
legend(axs(3), 'model', 'target', 'location', 'northeast')

if trainRNN % chi2 over training runs
    plot(axs(4), pVars(1 : nRun)) % only get a curve for training runs since nRunTrain > 1
    plot(axs(5), chi2(1 : nRun));
    set(axs(5), 'ylim', [-0.1 1]);
else % chi2 over each timestep of one test run through a trial
    plot(axs(5), chi2(1 : nSP_Data));
    set(axs(5), 'xlim', [1 nSP_Data]);
end

drawnow;

end