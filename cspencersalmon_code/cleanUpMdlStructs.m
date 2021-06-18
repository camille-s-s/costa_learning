clearvars
close all

% in and outdirs
bd              = '~/Dropbox (BrAINY Crew)/costa_learning/';
mdlDir          = [bd 'models/'];
RNNfigdir       = [bd 'models/figures/'];
spikeInfoPath   = [bd 'reformatted_data/'];


%%
addpath(genpath(bd))

cd(mdlDir)
mdlFiles = dir('rnn_*_set*_trial*.mat');


allSetIDs = unique(arrayfun(@(i) ...
    str2double(mdlFiles(i).name(strfind(mdlFiles(i).name,'set') + 3 : strfind(mdlFiles(i).name,'trial') - 2)), 1:length(mdlFiles)));
allTrialIDs = unique(arrayfun(@(i) ...
    str2double(mdlFiles(i).name(strfind(mdlFiles(i).name,'trial') + 5 : end - 4)), 1:length(mdlFiles)));

for i = 1:length(mdlFiles)
    
    mdlName = mdlFiles(i).name;
    mdlPath = [mdlDir, mdlName];
    load(mdlPath)
    
    % why keep the full RNN it's huge
    if isfield(RNN.mdl, 'RNN')
        RNN.mdl = rmfield(RNN.mdl, 'RNN');
        save(mdlPath, 'RNN')
    end
    
    setID = str2double(mdlName(strfind(mdlName, 'set') + 3 : strfind(mdlName, 'trial') - 2));
    iTrl = str2double(mdlName(strfind(mdlName, 'trial')+5:end-4));
    
    % remove params struct for all sets aside from set 0
    if setID ~= 0 && isfield(RNN.mdl, 'params') % only 0th set gets params! why repeat!
        RNN.mdl = rmfield(RNN.mdl, 'params');
        save(mdlPath, 'RNN')
    end
    
    % but add these back in for indexing
    if ~isfield(RNN.mdl, 'iTrl')
        RNN.mdl.iTrl = iTrl;
        save(mdlPath, 'RNN')
    end
    
    if ~isfield(RNN.mdl, 'setID')
        RNN.mdl.setID = setID;
        save(mdlPath, 'RNN')
    end
    
    % fix params struct for set 0
    if setID == 0 && length(RNN.mdl.params) ~= 1% fix params which somehow got repeated
        arrayUnit = {{RNN.mdl.params.arrayUnit}'};
        arrayRegions = RNN.mdl.params(1).arrayRegions;
        RNN.mdl.params = rmfield(RNN.mdl.params, 'arrayUnit');
        RNN.mdl.params = rmfield(RNN.mdl.params, 'arrayRegions');
        allParams = fieldnames(RNN.mdl.params);
        
        for j = 1:length(allParams)
            try
                tmp = [RNN.mdl.params.(allParams{j})];
                assert(length(unique(tmp)) == 1)
            catch
                tmp = {RNN.mdl.params.(allParams{j})};
                assert(length(unique(cellfun(@char, tmp, 'un', 0))) == 1)
                
            end
        end
        
        RNN.mdl.params = RNN.mdl.params(1);
        RNN.mdl.params.arrayUnit = arrayUnit;
        RNN.mdl.params.arrayRegions = arrayRegions;
        save(mdlPath, 'RNN')
        
    end
    
    % add spike info to set 0
    if setID == 0 && ~isfield(RNN.mdl.params, 'spikeInfo')
        spikeInfoName = ['spikeData', mdlName(4 : median(strfind(mdlName, '_'))-1), '.mat'];
        load([spikeInfoPath, spikeInfoName], 'spikeInfo')
        RNN.mdl.params.spikeInfo = spikeInfo;
        save(mdlPath, 'RNN')
    end
    
end