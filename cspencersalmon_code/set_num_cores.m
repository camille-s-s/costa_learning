function set_num_cores(numcores)

if numcores>=36
    numcores=36; % exceeding 36 makes computations slower
end

% Select diff number of cores if using on server vs local node
pct_exists = which('gcp'); % local computer does not have PCT
if ~isempty(pct_exists)
    pp=gcp('nocreate'); % returns current pool if one exists
    if ~isempty(pp)
        delete(pp) % deletes it if one exists (so you can start a new one)
    end
    if strcmp(computer, 'GLNXA64')
        parpool('local',numcores);
    end
end

end