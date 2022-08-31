function [CURBD, CURBD_exc, CURBD_inh, trlAvgCURBD, trlAvgCURBD_exc, trlAvgCURBD_inh, curbdRgns, nRegionsCURBD] = costaCURBD(J_set, R_set, inds_set, curbdRgns, nTrls)
%% ONE J PER TRIAL VERSION
% NEED: curbdRgns, inds_set, J_set, R_set, minTrlsPerSet, tData_set

% loop along all bidirectional pairs of regions
nRegionsCURBD = sum(~cellfun(@isempty, curbdRgns(:, 2)));
curbdRgns = curbdRgns(~cellfun(@isempty, curbdRgns(:, 2)), :);

CURBD = cell(nRegionsCURBD, nRegionsCURBD);
CURBD_exc = cell(nRegionsCURBD, nRegionsCURBD);
CURBD_inh = cell(nRegionsCURBD, nRegionsCURBD);

trlAvgCURBD = cell(nRegionsCURBD, nRegionsCURBD);
trlAvgCURBD_exc = cell(nRegionsCURBD, nRegionsCURBD);
trlAvgCURBD_inh = cell(nRegionsCURBD, nRegionsCURBD);

CURBDLabels = cell(nRegionsCURBD, nRegionsCURBD);

if numel(unique(diff(inds_set))) == 1 % if working with truncated trials...
    calcAvgCURBD = true;
    trl_length = unique(diff(inds_set));
end

for iTarget = 1 : nRegionsCURBD
    
    in_target = curbdRgns{iTarget, 2};
    nUnitsTarget = numel(curbdRgns{iTarget, 2});
    
    if calcAvgCURBD
        P_full_mat = NaN(nUnitsTarget, trl_length, nTrls);
        P_exc_mat = NaN(nUnitsTarget, trl_length, nTrls);
        P_inh_mat = NaN(nUnitsTarget, trl_length, nTrls);
    end
    
    for iSource = 1 : nRegionsCURBD
        
        in_source = curbdRgns{iSource, 2};
        
        P_full_cat = NaN(numel(in_target), inds_set(nTrls + 1) - 1);
        P_exc_cat = NaN(numel(in_target), inds_set(nTrls + 1) - 1);
        P_inh_cat = NaN(numel(in_target), inds_set(nTrls + 1) - 1);

        for iTrl = 1 : nTrls % length(trlIDs)
            J_curbd = squeeze(J_set(:, :, iTrl)); % J from first sample point
            R_curbd = R_set(:, inds_set(iTrl) : inds_set(iTrl + 1) - 1); % R from ith to i+1th sample point

            % split by exc/inh
            J_exc = J_curbd; J_inh = J_curbd;
            J_exc(J_exc < 0 ) = 0; J_inh(J_inh > 0) = 0;
            
            P_full = J_curbd(in_target, in_source) * R_curbd(in_source, :);
            P_exc = J_exc(in_target, in_source) * R_curbd(in_source, :);
            P_inh = J_inh(in_target, in_source) * R_curbd(in_source, :);
            
            % separate CURBD on each trial in the set
            P_full_cat(:, inds_set(iTrl) : inds_set(iTrl + 1) - 1) = P_full;
            P_exc_cat(:, inds_set(iTrl) : inds_set(iTrl + 1) - 1) = P_exc;
            P_inh_cat(:, inds_set(iTrl) : inds_set(iTrl + 1) - 1) = P_inh;
            
            % nUnitsTarget x trlLength x minTrlsPerSet
            P_full_mat(:, :, iTrl) = P_full;
            P_exc_mat(:, :, iTrl) = P_exc;
            P_inh_mat(:, :, iTrl) = P_inh;
        end
        
        if calcAvgCURBD % calculate CURBD trial by trial, but average it for plotting
            assert(~any(isnan(P_full_mat(:))))
            trlAvgCURBD{iTarget, iSource} = mean(P_full_mat, 3);
            trlAvgCURBD_exc{iTarget, iSource} = mean(P_exc_mat, 3);
            trlAvgCURBD_inh{iTarget, iSource} = mean(P_inh_mat, 3);
        end
        
        if any(isnan(P_full_cat(:)))
            keyboard
        end
        
        CURBD{iTarget, iSource} = P_full_cat;% J_curbd(in_target, in_source) * R_curbd(in_source, :);
        CURBD_exc{iTarget, iSource} = P_exc_cat;
        CURBD_inh{iTarget, iSource} = P_inh_cat;
        CURBDLabels{iTarget, iSource} = [curbdRgns{iSource, 1}(1 : end - 3) ' to ' curbdRgns{iTarget, 1}(1 : end - 3)];
    end
end

end