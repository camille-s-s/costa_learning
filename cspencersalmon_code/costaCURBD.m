function [CURBD, avgCURBD, curbdRgns, nRegionsCURBD] = costaCURBD(J_set, R_set, inds_set, curbdRgns, nTrls)

%% ONE J PER TRIAL VERSION
% NEED: curbdRgns, inds_set, J_set, R_set, minTrlsPerSet, tData_set

% loop along all bidirectional pairs of regions
nRegionsCURBD = sum(~cellfun(@isempty, curbdRgns(:, 2)));
curbdRgns = curbdRgns(~cellfun(@isempty, curbdRgns(:, 2)), :);
CURBD = cell(nRegionsCURBD, nRegionsCURBD);
avgCURBD = cell(nRegionsCURBD, nRegionsCURBD);
CURBDLabels = cell(nRegionsCURBD, nRegionsCURBD);

if numel(unique(diff(inds_set))) == 1 % if working with truncated trials...
    calcAvgCURBD = true;
    trl_length = unique(diff(inds_set));
end

for iTarget = 1 : nRegionsCURBD
    
    in_target = curbdRgns{iTarget, 2};
    nUnitsTarget = numel(curbdRgns{iTarget, 2});
    
    if calcAvgCURBD
        curbd_mat = NaN(nUnitsTarget, trl_length, nTrls);
    end
    
    for iSource = 1 : nRegionsCURBD
        
        in_source = curbdRgns{iSource, 2};
        
        curbd_cat = NaN(numel(in_target), inds_set(nTrls + 1) - 1);
        
        for iTrl = 1 : nTrls % length(trlIDs)
            J_curbd = squeeze(J_set(:, :, iTrl)); % J from first sample point
            R_curbd = R_set(:, inds_set(iTrl) : inds_set(iTrl + 1) - 1); % R from ith to i+1th sample point
            
            % separate CURBD on each trial in the set
            curbd_cat(:, inds_set(iTrl) : inds_set(iTrl + 1) - 1) = J_curbd(in_target, in_source) * R_curbd(in_source, :);
            
            % nUnitsTarget x trlLength x minTrlsPerSet
            curbd_mat(:, :, iTrl) = J_curbd(in_target, in_source) * R_curbd(in_source, :);
        end
        
        if calcAvgCURBD % calculate CURBD trial by trial, but average it for plotting
            assert(~any(isnan(curbd_mat(:))))
            avgCURBD{iTarget, iSource} = mean(curbd_mat, 3);
        end
        
        if any(isnan(curbd_cat(:)))
            keyboard
        end
        
        CURBD{iTarget, iSource} = curbd_cat;% J_curbd(in_target, in_source) * R_curbd(in_source, :);
        CURBDLabels{iTarget, iSource} = [curbdRgns{iSource, 1}(1 : end - 3) ' to ' curbdRgns{iTarget, 1}(1 : end - 3)];
    end
end

% % get clims by excluding anything above 95th or below 5th %ile
% if calcAvgCURBD
%     CURBD_tmp = cell2mat(avgCURBD);
%     xTks = round(linspace(1, trl_length, 5)); % these will be trial averaged sample points
% else
%     CURBD_tmp = cell2mat(CURBD);
%     trlBrks = tData_set(inds_set(1 : minTrlsPerSet));
%     xTks = tData_set(inds_set(1 : 2 : minTrlsPerSet + 1)); % round(linspace(tData_set(inds_set(1)), tData_set(inds_set(minTrlsPerSet + 1) - 1), 6));
% end
%
% CLims = mean(abs(prctile(CURBD_tmp(:), [5 95])));
%
% % plot heatmaps of currents
% figure('color', 'w');
% set(gcf, 'units', 'normalized', 'outerposition', [0.0275 0.0275 0.95 0.95])
% axLPos = linspace(0.01, 0.95 - (0.95/7), 7);
% axWPos = 0.95 / size(CURBD, 2);
% count = 1;
%
% for iTarget = 1 : size(CURBD, 1)
%
%     nUnitsTarget = numel(curbdRgns{iTarget, 2});
%
%     for iSource = 1 : size(CURBD, 2)
%
%         subplot(size(CURBD, 1), size(CURBD, 2), count);
%         hold all;
%         count = count + 1;
%
%         if ~calcAvgCURBD
%
%             imagesc(tData_set(inds_set(1) : inds_set(minTrlsPerSet + 1) - 1), 1 : nUnitsTarget, CURBD{iTarget, iSource});
%             axis tight;
%             hold on,
%
%             % demarcate trials
%             arrayfun(@(i) ...
%                 line(gca, [trlBrks(i) trlBrks(i)], get(gca, 'ylim'), ...
%                 'linestyle', '-', 'linewidth', 1, 'color', [0.2 0.2 0.2]), ...
%                 1 : length(trlBrks))
%
%             set(gca, 'xtick', xTks, 'xticklabel', xTks)
%
%         else
%             imagesc(1 : trl_length, 1 : nUnitsTarget, avgCURBD{iTarget, iSource});
%             axis tight;
%             hold on,
%             set(gca, 'xtick', xTks, 'xticklabel', num2str([(xTks * dtData) - dtData]', '%.1f'))
%         end
%
%         if iTarget == size(CURBD, 1) && iSource == 1
%             xlabel('time (s)', 'fontweight', 'bold');
%         end
%
%         set(gca, 'Box', 'off', 'TickDir', 'out', 'yticklabel', '', 'FontSize', 11);
%
%         if iSource == 1
%             ylabel([curbdRgns{iTarget, 1}(1 : end - 3), '(', num2str(nUnitsTarget), ')'], 'fontweight', 'bold');
%         end
%
%         axPos = get(gca, 'OuterPosition');
%         set(gca, 'Outerposition', [axLPos(iSource) axPos(2) axWPos, axPos(4)])
%
%         title([curbdRgns{iSource, 1}(1 : end - 3), ' > ', curbdRgns{iTarget, 1}(1 : end - 3)], 'fontweight', 'bold');
%     end
% end
%
% cm = brewermap(100,'*RdBu');
% allAx = findall(gcf, 'type', 'axes');
% arrayfun(@(iAx) set(allAx(iAx),'clim', [-CLims, CLims]), 1 : length(allAx))
% colormap(cm)
% close
%
% % plot means of currents
% figure('color', 'w');
% set(gcf, 'units', 'normalized', 'outerposition', [0.0275 0.0275 0.95 0.95])
% count = 1;
%
% for iTarget = 1 : size(CURBD, 1)
%     nUnitsTarget = numel(curbdRgns{iTarget, 2});
%     for iSource = 1 : size(CURBD, 2)
%
%         subplot(size(CURBD, 1), size(CURBD, 2), count);
%         hold all;
%         count = count + 1;
%
%         plot(1 : trl_length, mean(avgCURBD{iTarget, iSource}, 1), 'linewidth', 1.5, 'color', 'k')
%         line(gca, get(gca, 'xlim'), [0 0], 'linestyle', ':', 'linewidth', 1, 'color', [0.2 0.2 0.2])
%
%         axis tight;
% set(gca, 'box', 'off', 'tickdir', 'out', 'fontsize', 11)
% if ~calcAvgCURBD
%         set(gca, 'xticklabel', '', 'yticklabel', '');
% else
%             set(gca, 'xtick', xTks, 'xticklabel', num2str([(xTks * dtData) - dtData]', '%.1f'), 'yticklabel', '')
% end
%
%         if iTarget == size(CURBD, 1) && iSource == 1
%             xlabel('time (s)', 'fontweight', 'bold');
%         end
%
%         if iSource == 1
%             ylabel([curbdRgns{iTarget, 1}(1 : end - 3), '(', num2str(nUnitsTarget), ')'], 'fontweight', 'bold');
%         end
%
%         title([curbdRgns{iSource, 1}(1 : end - 3), ' > ', curbdRgns{iTarget, 1}(1 : end - 3)], 'fontweight', 'bold');
%     end
% end
%
% % cm = brewermap(100,'*RdBu');
% allAx = findall(gcf, 'type', 'axes');
% allYLim = cell2mat(arrayfun(@(iAx) get(allAx(iAx), 'ylim'), 1 : length(allAx), 'un', 0)');
% YMax = max(abs(allYLim(:)));
% arrayfun(@(iAx) set(allAx(iAx),'ylim', [-YMax, YMax]), 1 : length(allAx))
% colormap(cm)

end