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

end