% param sweep RNN


% tauRNN tauWN 10*10 possibilities, run with identical parameters
% assess pVar and chi2 as a function of each combination

% param sweep
logvals = [0.001,0.01,0.1,1];
pVarVals = NaN(length(logvals),length(logvals));
chi2Vals = NaN(length(logvals),length(logvals));

for R = 1:length(logvals)
    tauRNN = logvals(R);
    for W = 1:length(logvals)
        tauWN = logvals(W);
        
        % train Model RNN targeting these three regions
        multiRNN = CS_RNN(struct( ...
            'tauRNN',tauRNN, ...
            'tauWN', tauWN, ...
            'plotStatus', false, ...
            'saveMdl', false));
        
        pVarVals(W,R) = multiRNN.all(end).mdl.pVar;
        chi2Vals(W,R) = multiRNN.all(end).mdl.chi2(end);
    end
end

figure,
imagesc(pVarVals), xlabel('tauRNN'), ylabel('tauWN'), colorbar, colormap(cool), title('pVar')
set(gca,'fontweight','bold', ...
    'xtick',1:4,'xticklabel', logvals,'ytick',1:4,'yticklabel', logvals)

figure,
imagesc(chi2Vals), xlabel('tauRNN'), ylabel('tauWN'), colorbar, colormap(cool), title('chi2')
set(gca,'fontweight','bold', ...
    'xtick',1:4,'xticklabel', logvals,'ytick',1:4,'yticklabel', logvals)


% add save RNN option