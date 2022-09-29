pinky = '/Volumes/camille/Dropbox (BrAINY Crew)/costa_learning/models/PINKY_VERSION';
homedir = '/Users/camille/Dropbox (BrAINY Crew)/costa_learning/models/PINKY_VERSION';

% SYNTAX: system([command FROM_HERE TO_HERE])
% Note: no slash after FROM_HERE (aka FROM_HERE/) means you put the lowest
% director in FROM_HERE in TO_HERE

if isfolder(pinky)
    cd(pinky)
    sessionFolders = [dir('v*'); dir('w*')];
else
    disp('pinky disconnected...')
    return
end

for iFile = 4 : length(sessionFolders)
    
    sessionName = sessionFolders(iFile).name;
    pinkyPath = [pinky filesep sessionName];
    homePath = [homedir];
    tic
    disp(['Copying updated models from Pinky to home for ', sessionName, '...'])
    [S1, O1] = system(['rsync -r -t  --exclude ''WN'' ', '"', pinkyPath, '" "', homePath, '"']);

    toc
    

    if any(S1)
        keyboard
    else
        disp('Success!')
    end
    
end


