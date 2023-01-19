function plot_comparative_FR_hists_each_unit(nBins, nUnits, nPredict, max_count, lower_thresh, upper_thresh, Rsample, targets, fig_title, use_reservoir)
% for use with fit_costa_RNN_prediction
% CSS 2022
%
% nBins = 25;
% max_count = nsp_Data - n_pred_steps;
% Rsample = R(:, iMdlSample);
%

% set up
fig_title(strfind(fig_title, '_')) = ' ';
if exist('use_reservoir', 'var')
    if use_reservoir
        nUnits = size(targets, 1);
        nPredict = nUnits;
    end
else
    use_reservoir = false;
end
        
axDim = ceil(sqrt(nPredict));
halfAxDim = floor(axDim / 2);
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

% transform from a count into a proportion of trial sample point
hist_counts_targets = hist_counts_targets ./ max_count;
hist_counts_mdl = hist_counts_mdl ./ max_count;

% set up subplot
figure('color', 'w', 'Position', [150 10 1400 925]);
axs = arrayfun( @(i) subplot(axDim, axDim, i, 'NextPlot', 'add', 'box', 'off', 'tickdir', 'out', 'fontsize', 10, 'fontweight', 'bold'), 1 : nPredict);

% plot model, then real
arrayfun(@(iUnit) plot(axs(iUnit), hist_centers, hist_counts_mdl(iUnit, :), 'linewidth', 1.5), 1 : nPredict)
arrayfun(@(iUnit) plot(axs(iUnit), hist_centers, hist_counts_targets(iUnit, :), 'linewidth', 1.5), 1 : nPredict)

% formatting
arrayfun( @(iUnit) set(axs(iUnit), 'ylim', [0 0.5], 'xlim', [hist_edges(1) hist_edges(end)]), 1 : nPredict)
arrayfun( @(iUnit) title(axs(iUnit), num2str(iUnit), 'fontsize', 14, 'fontweight', 'bold', 'color', [0.2 0.6 0.2]), 1 : nPredict)
xlabel(axs(1), 'FR', 'fontweight', 'bold', 'fontsize', 12)
ylabel(axs(1), 'proportion', 'fontweight', 'bold', 'fontsize', 12)
legend(axs(1), 'model', 'target', 'location', 'northeast')

% figure title
text(axs(halfAxDim), 0.75 * range(get(axs(halfAxDim), 'xlim')), 1.5 * max(get(axs(halfAxDim), 'ylim')), fig_title, 'fontweight', 'bold', 'fontsize', 16)

end