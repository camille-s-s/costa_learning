function plot_comparative_FR_hists_pop(nBins, nUnits, nPredict, max_count, lower_thresh, upper_thresh, Rsample, targets, fig_title)
% for use with fit_costa_RNN_prediction
% CSS 2022
%
% nBins = 25;
% max_count = nsp_Data - n_pred_steps;
% Rsample = R(:, iMdlSample);
%

% set up
fig_title(strfind(fig_title, '_')) = ' ';


min_FR = min([min(Rsample(:)), min(targets(:))]);
max_FR = max([max(Rsample(:)), max(targets(:))]);

if min_FR < -1 || max_FR > 1
    disp('you got some weird FR values here!')
    keyboard
end

hist_edges = linspace(lower_thresh, upper_thresh, nBins + 1);
hist_centers = hist_edges(1 : end - 1) + (diff(hist_edges) ./ 2); 
hist_counts_targets = cell2mat(arrayfun(@(i) histcounts(targets(i, :), hist_edges), 1 : nUnits, 'un', 0)'); % nUnits x nBins
hist_counts_mdl = cell2mat(arrayfun(@(i) histcounts(Rsample(i, :), hist_edges), 1 : nUnits, 'un', 0)'); % nUnits x nBins

% get count over all units
hist_counts_targets_pop = sum(hist_counts_targets, 1);
hist_counts_mdl_pop = sum(hist_counts_mdl, 1);

% transform from a count into a proportion of trial sample points (normalize also by # units you're counting over!)
hist_counts_targets_pop = hist_counts_targets_pop ./ (nPredict * max_count);
hist_counts_mdl_pop = hist_counts_mdl_pop ./ (nPredict * max_count);

% set up plot
figure('color', 'w', 'Position', [500 200 700 500]);
set(gca, 'NextPlot', 'add', 'box', 'off', 'tickdir', 'out', 'fontsize', 10, 'fontweight', 'bold')

% plot model, then real
plot(hist_centers, hist_counts_mdl_pop, 'linewidth', 1.5)
plot(hist_centers, hist_counts_targets_pop, 'linewidth', 1.5)

% formatting
set(gca, 'ylim', [0 0.5], 'xlim', [hist_edges(1) hist_edges(end)])
xlabel(gca, 'FR', 'fontweight', 'bold', 'fontsize', 12)
ylabel(gca, 'proportion', 'fontweight', 'bold', 'fontsize', 12)
legend(gca, 'model', 'target', 'location', 'northeast')
title(gca, fig_title,  'fontweight', 'bold', 'fontsize', 16)

end