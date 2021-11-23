%% I. CODER VARIABLES (BEHAVIORAL)

% WORKFLOW

% novelty_####_MM-DD-YYYY.bhv ?> bhv_read.m, Coder_maker.m, ??> novelty_#_YYYYMMDD_coder_nip# 
% (behavioral files used in neurophysiology analyses), aka
% /Users/camille/Dropbox/costa_learning/data/coder files/novelty_v_20161103_coder_nip1.mat

% TRIAL EVENTS DESCRIPTION

% 1. central fixation (500 - 750 ms)    fixlock
% 2. options presented                  stimlock
% 3. choice saccade                     stimlock + srt (ms)
% 4. fixate choice (500 ms)             stimlock + srt + 500ms??
% 5. choice outcome                     juicelock

% VARIABLE DESCRIPTION

% fixlock (fixation time)
% stimlock (stim on time)
% juicelock (reward delivered)
% reward: got a reward or not
% rewardpt: reward prediction; same as reward of previous trial
% str: duration of saccade, start at stimlock so stimlock + srt = (time of) choice

% validtrls (valid trial numbers for correct and error trials)
% stimid: each stimulus' ID

% picid: location and order of the novel stimulus first appeared.
% choice: which stimulus has been chosen(1,2,3, which are stimulus' ID).
% chosenstim: the ID of chosen stimulus
% direction: direction of the chosen stimulus (1 - 6)
% orientation: orientation of the (chosen) stimulus triangle (up or down)
% trlssincenov: trial's No. in every series
% Qsa=IEV+FEV; Qt=FEV; %% ps. These kind of parameters are from our RL model.
% BHVout: variates extracted from behavioral file. (see details in slide3)

arrayfun(@(iTrl) subJ(rmElements) = NaN

% novel or not (trlssincenov), reward or not...

%% Ia. CODER.BHVout VARIABLES

% updated by Hua Tang, Feb 20, 2019

% 1: mname % monkey name (the order in the name pool)
% 2-7: absolute first Trial Start Time
% 8-13: absolute current Trial Start Time
% 14: un-break trial number 
% 15-17: rrprob_aval (reward probability)
% 18-20: timeinarray (the times a specific pic continue appeared)
% 21-23: picid (when & where the novel image appeared)
% 24: arraydex£¨??£©
% 25: trlsincenovel (trial number since novel image)
% 26: dropin_loc (novel trial' location )
% 27: dropinrewprob (novel trial' reward probability )
% 28-30: sort(rrprob_aval,2)
% 31: chosenovel (whether chosen the novel image)
% 32: cloc (Choice location based on scenario file)
% 33: crewprob ()
% 34: csvl (Choice saccade velocity)
% 35: csrt (Choice saccadic reaction time) (msec)
% 36: choice_em (which image has been chosen)
% 37: choice_eyelog (choice)
% 38: juice (reward)
% 39: nVisRepofChoice (how many times the chosen image has been shown)
% 40: stimos_t-stimem_t 
% 41: 22222222 
% 42: fixholdtime (fixation holding time)

%% Ib. CODER.BHVError

% 4: no fixation, did not start trial
% 3: break fixation, started trial but quit before options presented
% 1: no response, did not choose the stimulus
% 5: early response, chose stimulus but did not hold it for enough time


% TRIAL EVENTS DESCRIPTION

% 1. central fixation (500 - 750 ms)    ERROR 4 (before this event)
% 2. options presented                  ERROR 3 (before this event, during previous event)
% 3. choice saccade                     ERROR 1 (before this event, during previous event)
% 4. fixate choice (500 ms)             ERROR 5 (during this event)
% 5. choice outcome                     juicelock

%% DATA FILES USED/# NEURONS

% Waldo
% Date			NIP1        NIP2
% 20160205		259			328
% 20160210		359			370
% 20160211 		291			391
% 
% Voltaire
% Date			NIP1        NIP2
% 20161103		235			220
% 20161104		240			264
% 20161107 		335			363

%% NEUROPHYS DATA FILE VARIABLES(#YYYYMMDD_nip#.mat)

% Coder: behavior parameters
% Neurons: the information of each neuron 
%   col 1: monkey
%   col 2: array
%   col 3: date
%   col 4: nip
%   col 5: electrode
%   col 6: unit
%   col 7: port
% aTS: timestamp for all neurons
% spikeCount: spike count (period: [-2, 2] to cue; bin = 20ms, details in
% Bin), dim = #units x #trials x #bins
% spikeSDF: spike density function


%% GOAL

% 1. make script to pull out data we need
%       a. data we need:
%           no error trials (Coder.validtrls = find(Coder.BHVerror==0);)

%           for two monkeys, for each array, for nip1 and nip2 (what is that), get all units
%           by....
%               time, name, region, condition
% 2. plot FR over time, for each array, all indices with behavioral
% markers, both measures: the spikes and the spike-convolved function
% 3. export to numpy array


% init lookup table
trlInfoVarNames = {'trls_since_nov_stim', ... % trlssincenov: #trials since novel stimulus, ...                                                                         NO 
    'nov_stim_idx', ... % picid: (when & where the novel image appeared)                                                                                                PROB NOT
    'nov_stim_loc', ... % dropin_loc (novel trial' location )                                                                                                           PROB NOT
    'nov_stim_rwd_prob', ... % dropinrewprob (novel trial' reward probability )                                                                                         YES
    'stimID', ... % stim: stimID (it's an actual value but TO DO: find out what the actual value means)                                                                 PROB NOT
    'reward_probs', ... % reward probability                                                                                                                            YES
    'choice_idx', ... % choices: which stim was chosen, will be 1, 2, or 3 (it's an index) PROB NOT unless use reward_probs(choice_idx) = chosen_stim_rwd_prob???
    'chosen_stim_orientation', ... % orientation: orientation of the (chosen) stimulus triangle (up or down) TO DO: IS THIS FOR THE CHOSEN STIM OR WHAT???              PROB NOT
    'chosen_stim_dir', ... % direction: direction of the chosen stimulus (1 - 6)                                                                                        NO
    'chosen_stim', ... % chosenstim: Coder.stim(Coder.choices) = Coder.chosenstim (or, stimID(choice_idx) = chosen_stim)
    'chose_nov_stim', ... % chosenovel (whether chose the novel image) YES
    'reward', ... % reward: got a reward or not YES
    'event_times', ... % fixlock, stimlock, choice time, juicelock NO
    'aligned_event_times'}; % the above but with lockEventTime subtracted NO 
 
% is it novel?
% trials since novel?
% value of best alternative?
% value of worst alternative? 
% is it rewarded?
% READ CHOICE BEHAVIOR SECTION
% expected value of novel option?
% how quant opportunity cost? (empirical value (how get that) of best alternative option)....


rewardProbChosenStim = arrayfun(@(t) T.reward_probs(t, T.choice_idx(t)), 1:nTrls)';
% if 0 in col1,  next 3 col must be 0, 0, 999
% if chose_nov_stim, chosen_stim must == nov_stim_idx

% maybe get value of chosen stim????
% units by time
% unit name, region, condition
% 
% PLOTTING: firing rate over time, all indices
% 
% 4 or 5 plots - one for each array with firing rate over time with behavioral markers
% 	two measures - spikes and spike-convolved function
%  three measures on rows, four arrays on columns ...mean(spikesSDF, [1
%  2]), mean(spikeCount, [1 2]), mean alignedSpikes, [1 2])
% 
% 
% export to numpy array 