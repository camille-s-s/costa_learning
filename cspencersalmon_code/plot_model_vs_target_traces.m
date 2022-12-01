function plot_model_vs_target_traces(targets, iTarget, nPlasticUnits, R, iModelSample)

if isequal(size(R, 2), size(targets, 2))
    R_plot = R(iTarget, :);
else
    R_plot = R(iTarget, iModelSample);
end

targets_plot = targets(iTarget, :);
    
% set up targets vs outputs for plotting
targets_plot_offset = cell2mat(arrayfun(@(i) targets_plot(i, :) + i, 1 : nPlasticUnits, 'un', 0)');
R_plot_offset = cell2mat(arrayfun(@(i) R_plot(i, :) + i, 1 : nPlasticUnits, 'un', 0)');

% set up figure
fig = figure('color', 'w', 'Position', [50 50 900 950]);
ax = axes('NextPlot', 'add');
axs = arrayfun( @(i) subplot(1, 2, i, 'NextPlot', 'add', 'box', 'off', 'tickdir', 'out', 'fontsize', 13, 'fontweight', 'bold'), [1 2]);
arrayfun( @(i) set(axs(i), 'ylim', [0 nPlasticUnits + 1]), [1 2])
arrayfun( @(i) set(axs(i), 'xlim', [0 size(targets_plot, 2) + 1]), [1 2])

% plot
arrayfun(@(i) plot(axs(1), targets_plot_offset(i, :), 'linewidth', 1), 1 : nPlasticUnits)
arrayfun(@(i) plot(axs(2), R_plot_offset(i, :), 'linewidth', 1), 1 : nPlasticUnits)

% label
xlabel(axs(1), 'time'), ylabel(axs(1), 'target units'), title(axs(1), 'TARGETS')
xlabel(axs(2), 'time'), ylabel(axs(2), 'model units'), title(axs(2), 'MODEL')
end
