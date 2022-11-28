function [rgns, badRgn, newOrder, nUnitsAll, JLblPos, JLinePos] = reorder_units_by_region(arrayRgns, desiredOrder, min_n_units_to_plot)
% set up indexing vectors for submatrices
% inputs: desiredOrder, arrayRgns from exp_data, min_n_units_to_plot (threshold)
% outputs: rgns, badRgn?, newOrder, nUnitsAll, JLblPos, JLinePos, newRgnInds?

% INPUTS
% desiredOrder = {'left_cdLPFC', 'right_cdLPFC', ...
%     'left_mdLPFC', 'right_mdLPFC', ...
%     'left_vLPFC', 'right_vLPFC', ...
%     'left_rdLPFC','right_rdLPFC'};
% min_n_units_to_plot = 5;
%
% OUTPUTS
%
% rns: re-ordered to fixed order across sessions and made more legible in plots
% badRgn: logical for removing regions without enough units according to min_n_units_to_plot
% newOrder: indexing vector for a given session to order its regions by desiredOrder
% nUnitsAll: unit counts for each region
% JLblPos: position for labels separating submatrices in plots
% JLinePos: position for lines separating submatrices in plots
%
%% 

rgns            = arrayRgns; % currentParams.arrayRegions; % or RNN.params.arrayRegions
arrayList       = rgns(:, 2);
nRegions        = length(arrayList);

% rearrange region order
rgnOrder = arrayfun(@(iRgn) find(strcmp(rgns(:,1), desiredOrder{iRgn})), 1 : nRegions); % ensure same order across sessions
rgns = rgns(rgnOrder, :);

% make more legible
rgns(:, 1) = arrayfun(@(iRgn) strrep(rgns{iRgn, 1}, 'left_', 'L'), 1 : nRegions, 'un', 0)';
rgns(:, 1) = arrayfun(@(iRgn) strrep(rgns{iRgn, 1}, 'right_', 'R'), 1 : nRegions, 'un', 0)';
rgns(:, 1) = arrayfun(@(iRgn) strrep(rgns{iRgn, 1}, 'LPFC', ''), 1 : nRegions, 'un', 0)';

% rgnLabels = rgns(:, 1);
% rgnIxToPlot = cell2mat(arrayfun(@(iRgn) contains(rgnLabels{iRgn}(1), 'L'), 1 : nRegions, 'un', 0));
inArrays = rgns(:, 3); % ID bad rgns
badRgn = arrayfun(@(iRgn) sum(inArrays{iRgn}) < min_n_units_to_plot, 1 : nRegions); % if array has too few units, exclude it

% rearrange unit order according to new region order - resulting newOrder makes it so that within-rgn is on-diagonal and between-rgn is off-diagonal
nUnitsAll = NaN(nRegions, 1);
newOrder = [];

for iRgn = 1 : nRegions
    in_rgn = rgns{iRgn,3};
    newOrder = [newOrder; find(in_rgn)]; % reorder J so that rgns occur in order
    nUnitsRgn = sum(in_rgn);
    nUnitsAll(iRgn) = nUnitsRgn;
end
nUnits = sum(nUnitsAll);

% get inds for new order by region
tmp = [0; nUnitsAll];
JLblPos = arrayfun(@(iRgn) sum(tmp(1 : iRgn-1)) + tmp(iRgn)/2, 2 : nRegions); % for the labels separating submatrices
JLinePos = cumsum(nUnitsAll)'; % for the lines separating regions
% newRgnInds = [0, JLinePos];

end
