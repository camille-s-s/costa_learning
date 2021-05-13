% data_dir = '/Users/mattperich/Dropbox (BrAINY Crew)/CURBD/Datasets/NHP/data/';
% model_dir = '/Users/mattperich/Dropbox (BrAINY Crew)/CURBD/Datasets/NHP/models/';
model_dir = '/Users/camille/Dropbox (BrAINY Crew)/csRNN/data/from_matt/';
monkeys = {'D'};

master_td = [];
% master_td = trial_data;
for iRun = 1 %:5
    for iMonk = 1:length(monkeys)
        monkey = monkeys{iMonk};
        
        load(fullfile(model_dir,['rnn_fit' monkey '_perc100_run' num2str(iRun) '.mat']));
        
        
        R_ds = zeros(size(R,1),length(tData));
        for ii = 1:size(R,1)
            R_ds(ii,:) = interp1(t,R(ii,:),tData);
        end
        
        Ra = R_ds(in_AMY,:);
        Rs = R_ds(in_SC,:);
        Rv = R_ds(in_VS,:);
        
        
        % compute full currents
        Jsa = J(in_AMY,in_SC);
        Jva = J(in_AMY,in_VS);
        Jaa = J(in_AMY,in_AMY);
        Psa = Jsa * Rs;
        Pva = Jva * Rv;
        Paa = Jaa * Ra;
        
        Jss = J(in_SC,in_SC);
        Jvs = J(in_SC,in_VS);
        Jas = J(in_SC,in_AMY);
        Pss = Jss * Rs;
        Pvs = Jvs * Rv;
        Pas = Jas * Ra;
        
        Jsv = J(in_VS,in_SC);
        Jvv = J(in_VS,in_VS);
        Jav = J(in_VS,in_AMY);
        Psv = Jsv * Rs;
        Pvv = Jvv * Rv;
        Pav = Jav * Ra;
        
        % compute inhibitory currents
        J_inh = J;
        J_inh(J_inh > 0) = 0;
        Jsa_inh = J_inh(in_AMY,in_SC);
        Jva_inh = J_inh(in_AMY,in_VS);
        Jaa_inh = J_inh(in_AMY,in_AMY);
        Psa_inh = Jsa_inh * Rs;
        Pva_inh = Jva_inh * Rv;
        Paa_inh = Jaa_inh * Ra;
        
        Jss_inh = J_inh(in_SC,in_SC);
        Jvs_inh = J_inh(in_SC,in_VS);
        Jas_inh = J_inh(in_SC,in_AMY);
        Pss_inh = Jss_inh * Rs;
        Pvs_inh = Jvs_inh * Rv;
        Pas_inh = Jas_inh * Ra;
        
        Jsv_inh = J_inh(in_VS,in_SC);
        Jvv_inh = J_inh(in_VS,in_VS);
        Jav_inh = J_inh(in_VS,in_AMY);
        Psv_inh = Jsv_inh * Rs;
        Pvv_inh = Jvv_inh * Rv;
        Pav_inh = Jav_inh * Ra;
        
        % compute excitatory currents
        J_exc = J;
        J_exc(J_exc < 0) = 0;
        Jsa_exc = J_exc(in_AMY,in_SC);
        Jva_exc = J_exc(in_AMY,in_VS);
        Jaa_exc = J_exc(in_AMY,in_AMY);
        Psa_exc = Jsa_exc * Rs;
        Pva_exc = Jva_exc * Rv;
        Paa_exc = Jaa_exc * Ra;
        
        Jss_exc = J_exc(in_SC,in_SC);
        Jvs_exc = J_exc(in_SC,in_VS);
        Jas_exc = J_exc(in_SC,in_AMY);
        Pss_exc = Jss_exc * Rs;
        Pvs_exc = Jvs_exc * Rv;
        Pas_exc = Jas_exc * Ra;
        
        Jsv_exc = J_exc(in_VS,in_SC);
        Jvv_exc = J_exc(in_VS,in_VS);
        Jav_exc = J_exc(in_VS,in_AMY);
        Psv_exc = Jsv_exc * Rs;
        Pvv_exc = Jvv_exc * Rv;
        Pav_exc = Jav_exc * Ra;
        
        
        % add them to trial data struct
        count = 0;
        for trial = 1:length(trial_data)
            trial_data(trial).run = iRun;
            
            N = size(trial_data(trial).AMY_spikes,1);
            idx = count + (1:N);
            
            % add full currents
            trial_data(trial).Curr_ScAmy = Psa(:,count + (1:N))';
            trial_data(trial).Curr_VsAmy = Pva(:,count + (1:N))';
            trial_data(trial).Curr_AmyAmy = Paa(:,count + (1:N))';
            
            trial_data(trial).Curr_ScSc = Pss(:,count + (1:N))';
            trial_data(trial).Curr_VsSc = Pvs(:,count + (1:N))';
            trial_data(trial).Curr_AmySc = Pas(:,count + (1:N))';
            
            trial_data(trial).Curr_ScVs = Psv(:,count + (1:N))';
            trial_data(trial).Curr_VsVs = Pvv(:,count + (1:N))';
            trial_data(trial).Curr_AmyVs = Pav(:,count + (1:N))';
            
            % add inhibitory currents
            trial_data(trial).Curr_ScAmy_inh = Psa_inh(:,count + (1:N))';
            trial_data(trial).Curr_VsAmy_inh = Pva_inh(:,count + (1:N))';
            trial_data(trial).Curr_AmyAmy_inh = Paa_inh(:,count + (1:N))';
            
            trial_data(trial).Curr_ScSc_inh = Pss_inh(:,count + (1:N))';
            trial_data(trial).Curr_VsSc_inh = Pvs_inh(:,count + (1:N))';
            trial_data(trial).Curr_AmySc_inh = Pas_inh(:,count + (1:N))';
            
            trial_data(trial).Curr_ScVs_inh = Psv_inh(:,count + (1:N))';
            trial_data(trial).Curr_VsVs_inh = Pvv_inh(:,count + (1:N))';
            trial_data(trial).Curr_AmyVs_inh = Pav_inh(:,count + (1:N))';
            
            % add excitatory currents
            trial_data(trial).Curr_ScAmy_exc = Psa_exc(:,count + (1:N))';
            trial_data(trial).Curr_VsAmy_exc = Pva_exc(:,count + (1:N))';
            trial_data(trial).Curr_AmyAmy_exc = Paa_exc(:,count + (1:N))';
            
            trial_data(trial).Curr_ScSc_exc = Pss_exc(:,count + (1:N))';
            trial_data(trial).Curr_VsSc_exc = Pvs_exc(:,count + (1:N))';
            trial_data(trial).Curr_AmySc_exc = Pas_exc(:,count + (1:N))';
            
            trial_data(trial).Curr_ScVs_exc = Psv_exc(:,count + (1:N))';
            trial_data(trial).Curr_VsVs_exc = Pvv_exc(:,count + (1:N))';
            trial_data(trial).Curr_AmyVs_exc = Pav_exc(:,count + (1:N))';
            count = count + N;
        end
        
        
        %
        curr_list = { ...
            'AMY_spikes', ...
            'SC_spikes', ...
            'VS_spikes', ...
            'Curr_AmyAmy', ...
            'Curr_ScAmy', ...
            'Curr_VsAmy', ...
            'Curr_AmySc', ...
            'Curr_ScSc', ...
            'Curr_VsSc', ...
            'Curr_AmyVs', ...
            'Curr_ScVs', ...
            'Curr_VsVs', ...
            'Curr_AmyAmy_inh', ...
            'Curr_ScAmy_inh', ...
            'Curr_VsAmy_inh', ...
            'Curr_AmySc_inh', ...
            'Curr_ScSc_inh', ...
            'Curr_VsSc_inh', ...
            'Curr_AmyVs_inh', ...
            'Curr_ScVs_inh', ...
            'Curr_VsVs_inh', ...
            'Curr_AmyAmy_exc', ...
            'Curr_ScAmy_exc', ...
            'Curr_VsAmy_exc', ...
            'Curr_AmySc_exc', ...
            'Curr_ScSc_exc', ...
            'Curr_VsSc_exc', ...
            'Curr_AmyVs_exc', ...
            'Curr_ScVs_exc', ...
            'Curr_VsVs_exc', ...
            };
        
        td = trial_data(1:4);
        for ii = 1:length(curr_list)
            td = dimReduce( ...
                softNormalize( ...
                smoothSignals(td, ...
                struct('signals',curr_list{ii},'width',0.01)), ...
                struct('signals',curr_list{ii},'alpha',0.01)), ...
                struct('signals',curr_list{ii},'num_dims',20));
        end
        master_td = catTDs(master_td,td);
    end
    close all;
end
% save(fullfile(data_dir,['master_td_runs.mat']),'master_td');
