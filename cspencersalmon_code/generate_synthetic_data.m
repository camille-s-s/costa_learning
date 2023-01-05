function synthetic_trials = generate_synthetic_data(training_inputs, unit_indices, n_waves, pct_jitter, trl_fraction, smoothWidth, dtData)
% make square wave using n_waves, wave_dur, wave_height
% INPUTS: training_inputs, n_waves, pct_jitter, trl_fraction, smoothWidth, dtData
% n_waves = 1;
% pct_jitter = 0.1; % jitter proportion (in proportion of current trial sample points)

% set up
data_dims = cell2mat(cellfun(@size, training_inputs, 'un', 0)); % nTrlsTrain x 1 cell. each element has dim nUnitsTrain x nsp_trial for that trial
n_trls = size(data_dims, 1);
n_units = unique(data_dims(:, 1)); assert(numel(n_units) == 1)
nsp_trial = data_dims(:, 2);

if ~exist('unit_indices', 'var') || isempty(unit_indices)
    unit_indices = 1 : n_units;
else
    n_replicates = unique(diff(unit_indices)); % should be a scalar
end

% choose wave height. initially, want same height all units all trials (a proportion of mean_train_FR_all)
inputs_mat = cell2mat(training_inputs');
mean_FR = mean(inputs_mat, 2);
std_FR = std(inputs_mat, [], 2);
mean_FR_all = mean(mean(inputs_mat, 2));
wave_height = (1 / trl_fraction) * mean_FR_all; 

% choose wave dur. initially, want mean FR synthetic trial == mean_train_FR_all, same dur all units all trials
mean_sp = mean(nsp_trial);
wave_dur_in_trl = round((trl_fraction) * mean_sp);
if any(rem(wave_dur_in_trl, 2)) % if odd
    half_wave_dur = (wave_dur_in_trl - 1) / 2; % wave pos is gonna be [half_wave, wave_pos, half_wave]
else % otherwise if even
    half_wave_dur = wave_dur_in_trl / 2; % wave pos is gonna be [half_wave, wave_pos, half_wave - 1]
end

% random placements with fixed random jitter of wave within trial for each unit. for each unit, this is a fixed proportion of trial dur
wave_locs = arrayfun( @(iUnit) randperm(100, n_waves), 1 : length(unit_indices))' ./ 100; % initially, same loc each unit all trials

% now loop through trials!
synthetic_trials = cell(n_trls, 1);
for iTrl = 1 : n_trls
    X = training_inputs{iTrl}(unit_indices, :); % not used yet but will if/when more complicated versions of this
    synth_X = zeros(length(unit_indices), 3 * nsp_trial(iTrl)); % padding equal to one trial length on either side for smoothing
    
    % generate new fixed random jitters, one for each unit, each trial (to make N random numbers in interval (a, b) with: r = a + (b - a) .* rand(N, 1))
    wave_jitters = -pct_jitter + (pct_jitter - -pct_jitter) .* rand(length(unit_indices), 1);
    
    % convert wave locations and jitters into indices (next four lines are all n_units x )
    wave_pos_in_trl = round((wave_locs + wave_jitters) * nsp_trial(iTrl));
    
    % assign wave at dur and pos to all units at once in this trial
    if any(rem(wave_dur_in_trl, 2)) % if odd
        wave_inds_in_trl = [wave_pos_in_trl - half_wave_dur, wave_pos_in_trl + half_wave_dur];
    else % if even
        wave_inds_in_trl = [wave_pos_in_trl - half_wave_dur, wave_pos_in_trl + half_wave_dur - 1];
    end
    
    % get locations as subscripts to put wave into
    wave_subs = cell2mat(arrayfun( @(iUnit) [repmat(iUnit, wave_dur_in_trl, 1), (wave_inds_in_trl(iUnit, 1) : wave_inds_in_trl(iUnit, 2))'], 1 : length(unit_indices), 'un', 0)');
    wave_subs(:, 2) = wave_subs(:, 2) + nsp_trial(iTrl); % adjust indexing for padding
    synth_X(sub2ind(size(synth_X), wave_subs(:, 1), wave_subs(:, 2))) = wave_height; % put in wave!
    
    % smooth with same parameters as for actual data
    synth_data_smoothed = smooth_data(synth_X', dtData, smoothWidth); synth_data_smoothed = synth_data_smoothed'; % try direct convolution custom not matlab toolbox
    synthetic_data = synth_data_smoothed(:, nsp_trial(iTrl) + 1 : 2 * nsp_trial(iTrl)); % trim the fat
    
    % unit_indices goes with use_reservoir which means we need to make replicates of rows!
    if ~isempty(unit_indices) && ~isequal(unit_indices, 1 : n_units) 
        synthetic_data = cell2mat(arrayfun(@(iUnit) repmat(synthetic_data(iUnit, :), n_replicates, 1), 1 : length(unit_indices), 'un', 0)');
    end
    
    % stick it in the cell for all trials
    synthetic_trials{iTrl} = synthetic_data;
end
end




