function [inputWN] = get_frozen_input_WN(nUnits, ampWN, tauWN, ampInWN, nsp_RNN, dtRNN)
% set up white noise inputs (from CURBD) filtered and spatially delocalized
% WN that is frozen (eqn 5 in CURBD) where iWN is a random variable drawn
% from Gaussian dist with mean 0 and unit variance, parameters h_0 = 1 and
% tau_WN = 0.1 controlling scale and correlation time respectively
%
% ampWN:                sqrt( tauWN / dtRNN )
% nUnits:               # units in RNN
% nsp_RNN:              # sample points in RNN
% dtRNN:                dtData / dtFactor; time step (in s) for integration
% tauWN:                decay constant on filtered white noise inputs
% ampInWN:              input amplitude of filtered white noise
%

iWN = ampWN * randn( nUnits, nsp_RNN );
inputWN = ones(nUnits, nsp_RNN);

for t = 2 : nsp_RNN
    inputWN(:, t) = iWN(:, t) + (inputWN(:, t - 1) - iWN(:, t)) * exp( -(dtRNN / tauWN) );
end

inputWN = ampInWN * inputWN;

end