function [Qsa,Qtran] = mdpChoice_appx_ratio_v3_PDPorSER(S,R,cho,rew,picid,runstate,picklambda)

fitModel     = 0;
improveModel = 0;

[prm, ~] = estimateUtility(fitModel, improveModel,picklambda);

[Qsa, Qtran] = runNoveltyBlock(prm,S,R,cho,rew,picid,runstate);

%dq = Qtran(1:length(choices))-Qtran(length(choices)+1:2*length(choices));

%plotNovelty(rewards, choices, novelOption, Qsa, Qtran, prm);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% setup test data and run several random versions to see if value
%%% estimates make sense
function [prm, utility] = estimateUtility(fitModel, improveModel,picklambda)

prm = defineParameters();

if fitModel == 1 && improveModel == 1
    fprintf('Fitting and improving?');
    return;
end

if fitModel == 1
    
    [prm, utility, betav, XKmat] = fitModelParameters(prm);        
    
else
    
    prm.lambda = picklambda;
    %try
        ttxt = sprintf('mdpNovelty0_%d_%d_%d_%d_%.3f_%d_%d_%d.mat', prm.dn, max(prm.nValues), prm.Mx, prm.BOrder, prm.lambda, length(prm.nEval), prm.maxPwr, 0);
    %catch
    %    ttxt = sprintf('mdpNovelty0_%d_%d_%d_%d_%.3f_%d_%d_%d_%d.mat', prm.dn, max(prm.nValues), prm.Mx, prm.BOrder, prm.lambda, length(prm.nEval), prm.maxPwr, 0, 0);
    %end;
    
    disp(ttxt);
    disp(prm.lambda);
    load(ttxt);
    
    
    betav = betav;
    
end

if improveModel == 1
    
    prm.utility = utility;
    prm.XP = XPMat;
    prm.XKmat = XKmat;
    
    fprintf('Length of utility %d\n', length(utility));
    
    [utility, ~, betav] = valueIteration(prm);
    
    rssqr = var((utility - prm.XKmat*betav));
    fprintf('Residual %.4f\n', rssqr);
    tt
    ttxt = sprintf('mdpNovelty0_%d_%d_%d_%d_%.3f_%d_%d_%d.mat', prm.dn, max(prm.nValues), prm.Mx, prm.BOrder, prm.lambda, length(prm.nEval), prm.maxPwr, 0);

    save(ttxt, 'betav', 'utility', 'XPMat', 'XKmat');
    
end


prm.betav = betav;

prm.XKmat = XKmat;

predUtility = prm.XKmat*betav;

plot(utility, predUtility, '.', min(utility):0.2:max(utility), min(utility):0.2:max(utility));

rssqr = var((utility - prm.XKmat*betav));
fprintf('Residual %.4f\n', rssqr);

dmat = inv(XKmat'*XKmat + prm.weightPrior*eye(size(XKmat, 2)))*rssqr;

prm.s = sqrt(diag(dmat));

prm.t = prm.betav./prm.s;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [prm, utility, betav, XKmat] = fitModelParameters(prm)

XKmat = iterateBasis(prm);

fprintf('XKmat %d %d\n', size(XKmat));

prm.XP = (XKmat'*XKmat + prm.weightPrior*eye(size(XKmat, 2)))\XKmat';

XPMat = prm.XP;

ftxt = sprintf('projectionMat_Novelty_%d_%d_%d_%d_%d_%d_%d.mat', prm.dn, ...
    max(prm.nValues), prm.Mx, prm.BOrder, length(prm.nEval), prm.maxPwr, 0);

save(ftxt, 'XPMat', 'XKmat');

load(ftxt);

prm.XP = XPMat;
prm.XKmat = XKmat;

fprintf('projection matrix is %d by %d\n', size(prm.XP, 1), size(prm.XP, 2));

prm.utility = [];

[utility, ~, betav] = valueIteration(prm);

rssqr = var((utility - prm.XKmat*betav));
fprintf('Residual %.4f\n', rssqr);

ttxt = sprintf('mdpNovelty0_%d_%d_%d_%d_%.3f_%d_%d_%d.mat', prm.dn, ...
    max(prm.nValues), prm.Mx, prm.BOrder, prm.lambda, length(prm.nEval), prm.maxPwr, 0);

save(ttxt, 'betav', 'utility', 'XPMat', 'XKmat');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function prm = defineParameters()

prm.priorProb(1) = 1;
prm.priorProb(2) = 2;
prm.K       = 3;
% prm.lambda  = 0.995;
prm.lambda  = 0.9;
prm.nValues = 0 : 150;
prm.pValues = 0 : 0.5 : 1;
prm.periodic = 1; %%% controls whether final state is to zero trials, or to zero reward
prm.aperiodicTransform = 0;
prm.tau = 0.9; %%% for aperiodic transform

%%% interpolant basis
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
prm.Mx     = 3;  %%% defines basis
%%% 0 == pwr + log, -1 == logistic, -2 == power, -3 == power + sqrt,
%%%  -4 == pwr + logistic
prm.maxPwr = 2; %%% for polynomial
prm.BOrder = 2;
prm.dn = -5;
prm.weightPrior = 10^-5;
prm.dp = 0.3333333;
prm.nsi    = min(prm.nValues) : prm.dn : max(prm.nValues); %%% nodes
prm.nEval  = min(prm.nValues) : prm.dn : max(prm.nValues);

prm.psi    = min(prm.pValues) : prm.dp : max(prm.pValues); %%% nodes
prm.pEval  = min(prm.pValues) : prm.dp : max(prm.pValues);

if prm.dn < 0
    
    if prm.dn == -30
        prm.nsi   = [0 3 10 50 max(prm.nValues) 181];
        prm.nEval = [0 3 10 50 max(prm.nValues)];   
    elseif prm.dn == -20
        prm.nsi   = exp(0:5/3:5);
        prm.nEval = exp(0:5/3:5);   
        
        prm.psi    = [0 0.33333 0.666667 1]; %%% nodes
        prm.pEval  = [0 0.33333 0.666667 1];

    elseif prm.dn == -10
        prm.nsi   = exp(0:5/4:5);
        prm.nEval = exp(0:5/4:5);  
        
        prm.psi    = [0 : 0.25 : 1]; %%% nodes
        prm.pEval  = [0 : 0.25 : 1];
        
    elseif prm.dn == -5
        prm.nsi   = [0 150];
        prm.nEval = [exp(0:5/4:5)];  
        
        prm.psi    = [0 : 1]; %%% nodes
        prm.pEval  = [0 : 0.25 : 1];
        
    end
    
end

fprintf('Model: Mx %d Power %d Basis order %d dn %d maxr %d\n', prm.Mx, prm.maxPwr, prm.BOrder, prm.dn, max(prm.nValues));


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function plotNovelty(rewards, choices, novelOption, Qsa, Qtran, prm)

N = length(rewards);
figure;

subplot(2,2,1);
plot(1:N, Qsa(1:N), 1:N, Qsa(N+1:2*N), 1:N, Qsa(2*N+1:end));
axis([0 25 -Inf Inf]);
legend;

subplot(2,2,2);
plot(1:N, Qtran(1:N), 1:N, Qtran(N+1:2*N), 1:N, Qtran(2*N+1:end));    
axis([0 25 -Inf Inf]);

subplot(2,2,3);
plot(1:N, Qtran(1:N)-Qtran(N+1:2*N), 1:N, Qtran(1:N)-Qtran(2*N+1:end), ...
     1:N, Qtran(N+1:2*N) -Qtran(2*N+1:end));  
axis([0 25 -Inf Inf]);

subplot(2,2,4);
plot(1:N, rewards*2, '*', 1:N, choices(:,1)-1 + 0.1, 'x');    

hold on;

novelSum = NaN*ones(size(novelOption, 1), 1);
q1 = find(novelOption(:, 1) == 1);
novelSum(q1) = 1;
q2 = find(novelOption(:, 2) == 1);
novelSum(q2) = 2;
q3 = find(novelOption(:, 3) == 1);
novelSum(q3) = 3;

plot(1:N, novelSum, 'o', 1:N, 0.5*ones(N, 1));

axis([0 25 -Inf Inf]);


nSteps = 5;
fappx = zeros(nSteps, nSteps);
Chv = zeros(nSteps, nSteps, 3);

c3 = 10;
for c1 = 0 : nSteps
    for c2 = 0 : nSteps
    
        cstate = [0.5 0.5 0.5 c1 c2 c3];

        [Chv(c1+1, c2+1, :), Cht(c1+1, c2+1, :)] = actionValueNovel(prm, cstate, prm.betav);   
        
        fappx(c1+1, c2+1) = f_utility(prm, cstate, prm.betav);
        
    end
    
end

c2 = 5;
c1Ctr = 1;
for c1 = 0 : 0.1 : nSteps
    
    cstate = [0.5 0.5 0.5 c1 c2 c3];

    funcd(c1Ctr) = f_utility(prm, cstate, prm.betav);
    c1Ctr = c1Ctr + 1;
        
end

figure;
subplot(2,2,1);
plot(Chv(:, 1, 1));

c1 = 5;
c2 = 5;
c3 = 5;
c1Ctr = 1;

Rv = []

c1Ctr = 1;
for c1 = 1 : 5 : 16
    r1Ctr = 1;
    for r1 = 0 : 0.1 : 1

        cstate = [r1 0.5 0.5 c1 c2 c3];

        [Rv(r1Ctr, c1Ctr, :), ~] = actionValueNovel(prm, cstate, prm.betav);   

        r1Ctr = r1Ctr + 1;

    end
    
    c1Ctr = c1Ctr + 1;
end

subplot(2,2,2);
plot(Rv(:, :,1));

for c1 = 1 : 6
    
    cstate = [1 0 0 c1 0 0];
    
    [Zv(c1, :), ~] = actionValueNovel(prm, cstate, prm.betav);   
    
end

subplot(2,2,3);
plot(Zv);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% for running synthetic daata
function [Qsat, Qtrant] = runNoveltyBlock(prm,S,R,cvec,rvec,picid,runstate)

%runstate = 2;



nTrials = size(S,1);

Qsat   = zeros(nTrials, prm.K);
Qtrant = zeros(nTrials, prm.K);

S = S+prm.priorProb(2);
R = R+prm.priorProb(1);

if runstate == 1,
    parfor t = 1:nTrials

        cstate = [R(t,1:3)./S(t,1:3) S(t,1:3)];

        [Qsa, Qtran] = actionValueNovel(prm, cstate, prm.betav);

        Qsat(t, :) = Qsa;
        Qtrant(t, :) = Qtran;

    end;
elseif runstate == 2
    nTrials = size(cvec,1);
    R = zeros(nTrials,prm.K);
    for cn = 1:prm.K
        R(find(cvec==cn),cn) = rvec(find(cvec==cn));
    end;
    novelOption = picid;
    S = zeros(prm.K, 2);
    S(1:3, 1) = prm.priorProb(1);
    S(1:3, 2) = prm.priorProb(2);
    rewards = zeros(nTrials, 1);
    Qsat   = zeros(nTrials, prm.K);
    Qtrant = zeros(nTrials, prm.K);
    
    for t = 1 : nTrials

        z = find(novelOption(t, :) == 1);

        if sum(z) > 0
            S(z, 1) = prm.priorProb(1);
            S(z, 2) = prm.priorProb(2);

            novelTrial = t;

        end

        cstate = [(S(1:3, 1)./S(1:3, 2))' S(1:3, 2)'];

        qi = find(isnan(cstate) == 1);
        cstate(qi) = 0.5;

        [Qsa, Qtran] = actionValueNovel(prm, cstate, prm.betav);

        choicei = cvec(t);

        Qsat(t, :) = Qsa;
        Qtrant(t, :) = Qtran;

        rewards(t) = R(t, choicei);

        St(:,:,t) = S;
        S(choicei, 1) = S(choicei, 1) + R(t, choicei);
        S(choicei, 2) = S(choicei, 2) + 1;

        choices(t, 1) = choicei;    
            
    end;
end;
    
    

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% for running synthetic daata
function [aB] = defineBasis(M, xsi, Xeval, prm)

K = length(xsi);

if M > 0
    Bxsi = [xsi(1)*ones(1, M) xsi xsi(end)*ones(1, M)];
end

B = zeros(M, length(Xeval), K + 2*M - 1);

if M > 0

    aB = [];
    for mi = 1 : M
        for ti = M - mi + 1 : K + 2*M - mi

            if mi == 1
            
                if Bxsi(ti) - Bxsi(ti+1) == 0
                    B(mi, :, ti) = zeros(1, length(Xeval));
                else
                    if ti < M + K
                        B(mi, :, ti) = double((Xeval >= Bxsi(ti) & Xeval < Bxsi(ti+1))');
                    else
                        B(mi, :, ti) = double((Xeval >= Bxsi(ti) & Xeval <= Bxsi(ti+1))');                
                    end
                end
                
            else
                
                if (Bxsi(ti+mi-1) - Bxsi(ti) == 0) && (Bxsi(ti+mi) - Bxsi(ti+1) == 0)
                    B(mi, :, ti) = zeros(1, length(Xeval));
                elseif Bxsi(ti+mi) - Bxsi(ti+1) == 0
                    for xi = 1 : length(Xeval)
                        B(mi, :, ti) = B(mi-1, xi,   ti)*(Xeval(xi) -     Bxsi(ti))/(Bxsi(ti+mi-1) - Bxsi(ti));
                    end
                elseif (Bxsi(ti+mi-1) - Bxsi(ti) == 0)
                    for xi = 1 : length(Xeval)
                        B(mi, :, ti) = B(mi-1, xi, ti+1)*((Bxsi(ti+mi) - Xeval(xi))/(Bxsi(ti+mi) - Bxsi(ti+1)));                    
                    end
                else
                    for xi = 1 : length(Xeval)

                        B(mi, xi, ti) = B(mi-1, xi,   ti)*(Xeval(xi) -     Bxsi(ti))/(Bxsi(ti+mi-1) - Bxsi(ti)) + ...
                                        B(mi-1, xi, ti+1)*((Bxsi(ti+mi) - Xeval(xi))/(Bxsi(ti+mi) - Bxsi(ti+1)));

                    end
                end
            end                        
        end
        
        if length(Xeval) > 1
            aB = [aB squeeze(B(mi, :, M-mi+1:M+K))];
        else
            aB = [aB squeeze(B(mi, :, M-mi+1:M+K))'];
        end

    end
    
elseif M == 0
    
    aB = zeros(length(Xeval), prm.maxPwr);

    for pwr = 1 : prm.maxPwr + 1
        aB(:, pwr) = (Xeval.^(pwr-1))';
    end
    
    aB = [aB log(Xeval + 0.01)'];
    
elseif M == -1
    
    aB = zeros(length(Xeval), length(xsi));
    
    a = 1.5;

    for ti = 1 : length(xsi)
        aB(:, ti) = 1./(1 + exp(-a*(Xeval - xsi(ti))));
    end
    
elseif M == -2
    
    aB = zeros(length(Xeval), prm.maxPwr);

    for pwr = 1 : prm.maxPwr
        aB(:, pwr) = (Xeval.^(pwr))';
    end
    
elseif M == -3
    
    aB = zeros(length(Xeval), prm.maxPwr);

    for pwr = 1 : prm.maxPwr + 1
        aB(:, pwr) = (Xeval.^(pwr-1))';
    end
    
    aB = [aB sqrt(Xeval)'];
    
elseif M == -4
    
    aB = zeros(length(Xeval), prm.maxPwr);

    for pwr = 1 : prm.maxPwr + 1
        aB(:, pwr) = (Xeval.^(pwr-1))';
    end
    
    aB = [aB 1./(1 + exp(-Xeval))'];
    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [yhat, vX] = f_utility(prm, cstate, betav)

for di = 1 : prm.K

    xV(di, :) = defineBasis(prm.Mx, prm.psi, cstate(di), prm);

end

for di = 1 : prm.K

    xV(di+prm.K, :) = defineBasis(prm.Mx, prm.nsi, cstate(di+prm.K), prm);

end

vX = basisModelOrder(prm, xV, 1);

%vX = prm.XKmat;

if isempty(betav)
    yhat = [];
else
    yhat = vX*betav;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function XKmat = basisModelOrder(prm, aBx, pred)

if abs(prm.BOrder) == 1
    
    XKmat = [];
    
    for dimi = 1 : prm.K*2
        
        if pred == 0
            XKmat = [XKmat aBx];
        else
            XKmat = [XKmat aBx(dimi, :)];
        end
    end
            
elseif abs(prm.BOrder) == 2
    
    XKmat = [];
    
    for dimi1 = 1 : prm.K*2
        for dimi2 = dimi1 + 1 : prm.K*2
            
            if pred == 0
                XKmat = [XKmat kron(aBx, aBx)];
            else
                XKmat = [XKmat kron(aBx(dimi1, :), aBx(dimi2, :))];                    
            end
            
        end
    end 
    
    XKmat = [ones(size(XKmat, 1), 1) XKmat];
    
elseif abs(prm.BOrder) == 3
    
    XKmat = [];
    
    for dimi1 = 1 : prm.K*2
        for dimi2 = dimi1 + 1 : prm.K*2
            for dimi3 = dimi2 + 1 : prm.K*2
                
                if pred == 0
                    XKmat = [XKmat kron(aBx, kron(aBx, aBx))];
                else
                    XKmat = [XKmat kron(aBx(dimi3, :), kron(aBx(dimi2, :), aBx(dimi1, :)))];
                end
                
            end            
        end
    end     
    
elseif prm.BOrder == 6
    
    if pred == 0
    
        XKmat = aBx;

        for chi = 2 : prm.K*2

            XKmat = kron(aBx, XKmat);

        end 
        
    else
        
        XKmat = aBx(1, :);

        for chi = 2 : prm.K*2

            XKmat = kron(aBx(chi, :), XKmat);

        end 
        
    end
            
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function XKmat = iterateBasis(prm)

nEval = length(prm.nEval);
pEval = length(prm.pEval);

for ci = 1 : nEval
    bsn(ci, :) = defineBasis(prm.Mx, prm.nsi, prm.nEval(ci), prm);    
end

for ci = 1 : pEval
    bsp(ci, :) = defineBasis(prm.Mx, prm.psi, prm.pEval(ci), prm);    
end

bas = [bsp(1, :); bsp(1, :); bsp(1, :); ...
       bsn(1, :); bsn(1, :); bsn(1, :)];

tVec = basisModelOrder(prm, bas, 1);

XKmat = zeros(pEval, pEval, pEval, nEval, nEval, nEval, length(tVec));

for c1i = 1 : pEval
    
    for c2i = 1 : pEval
        
        for c3i = 1 : pEval
            
            for r1i = 1 : nEval
    
                for r2i = 1 : nEval

                    for r3i = 1 : nEval
                        
                        bas = [bsp(c1i, :); bsp(c2i, :); bsp(c3i, :); ...
                               bsn(r1i, :); bsn(r2i, :); bsn(r3i, :)];
                        
                        XKmat(c1i, c2i, c3i, r1i, r2i, r3i, :) = basisModelOrder(prm, bas, 1);
                                                
                    end
                end
                
            end
        end
    end    
end

XKmat = reshape(XKmat, (pEval^3)*(nEval^3), size(XKmat, 7));

% % XKmat = (XKmat - repmat(mean(XKmat), size(XKmat, 1), 1))./repmat(std(XKmat), size(XKmat, 1), 1);
% XKmat = (XKmat)./repmat(std(XKmat), size(XKmat, 1), 1);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [utility, Qsi, betav] = valueIteration(prm)

epsilon = 0.0001;

if ~isempty(prm.utility)
    epsilon = epsilon/1000;
    utility = prm.utility;
else
    utility = 30 + rand(size(prm.XP, 2), 1);    
end

Qsi = zeros(size(prm.XP, 2), 1);

betav = prm.XP*utility;

maxIter = 10000;
actionDistance = zeros(maxIter, 1);

maxSpan = zeros(maxIter, 1);
minSpan = zeros(maxIter, 1);

for iteri = 1 : maxIter
    
    [utility_update, Qsi_update] = stateUtilityPar(prm, betav);
    
    betav_update = prm.XP*utility_update;
    
    dfbetav = sum(abs(betav_update - betav));
    
    qi = find(utility_update > 0);
    dfUtil = abs(utility_update - utility);
      
    maxSpan(iteri) = max(dfUtil(qi));
    minSpan(iteri) = min(dfUtil(qi));

    if prm.lambda < 1
        if ((sum(dfbetav) < epsilon) | (maxSpan(iteri) - minSpan(iteri) < epsilon))
            break;
        end
    else
        
        if maxSpan(iteri) - minSpan(iteri) < epsilon
            break;
        end
               
    end
        
    actionDistance(iteri) = sum(abs(Qsi_update - Qsi));
        
    fprintf('iter %d mxSpan %.3f mnSpan %.3f Qsa dis %.2f df betav %.5e %.4f %.3f\n', iteri, ...
        maxSpan(iteri), minSpan(iteri), actionDistance(iteri), dfbetav, sum(dfUtil), max(utility_update));
    
    utility = utility_update;  
    betav = betav_update;
    Qsi = Qsi_update;
    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [utility_t, Qsi] = stateUtilityPar(prm, betav)

nEval = length(prm.nEval);
pEval = length(prm.pEval);

Qsi_b     = zeros(pEval, pEval, pEval, nEval, nEval, nEval);
utility_b = zeros(pEval, pEval, pEval, nEval, nEval, nEval);
         
parfor c1i = 1 : nEval
    
    blockMNI = zeros(pEval, pEval, pEval, nEval, nEval);
    blockQsi = zeros(pEval, pEval, pEval, nEval, nEval);

    for c2i = 1 : nEval

        for c3i = 1 : nEval
            
            for r1i = 1 : pEval
                
                if prm.nEval(c1i) == 0 & prm.pEval(r1i) > 0
                    continue;
                end
    
                for r2i = 1 : pEval
                    
                    if prm.nEval(c2i) == 0 & prm.pEval(r2i) > 0
                        continue;
                    end

                    for r3i = 1 : pEval
                        
                        if prm.nEval(c3i) == 0 & prm.pEval(r3i) > 0
                            continue;
                        end
                        
                        c1 = prm.nEval(c1i);
                        c2 = prm.nEval(c2i);
                        c3 = prm.nEval(c3i);
                        p1 = prm.pEval(r1i);
                        p2 = prm.pEval(r2i);
                        p3 = prm.pEval(r3i);
                        
                        cstate = [p1 p2 p3 c1 c2 c3];

                        [u, ~] = actionValueNovel(prm, cstate, betav);

                        [mxv, mxi] = max(u);
                        
                        blockMNI(r1i, r2i, r3i, c2i, c3i) = mxv;
                        blockQsi(r1i, r2i, r3i, c2i, c3i) = mxi;     
                        
                    end
                end
                
            end
        end
    end
    
    Qsi_b(:, :, :, c1i, :, :)    = blockQsi;
    utility_b(:, :, :,c1i, :, :) = blockMNI;
    
end

Qsi = reshape(Qsi_b, (pEval^3)*(nEval^3), 1);

utility_t = reshape(utility_b, (pEval^3)*(nEval^3), 1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Qsa, Qtran] = actionValueNovel(prm, cstate, betav)

r1 = cstate(1)*cstate(4);
r2 = cstate(2)*cstate(5);
r3 = cstate(3)*cstate(6);

c1 = cstate(4);
c2 = cstate(5);
c3 = cstate(6);

% rp(1) = (r1+prm.priorProb(1))/(c1+prm.priorProb(2));
% rp(2) = (r2+prm.priorProb(1))/(c2+prm.priorProb(2));
% rp(3) = (r3+prm.priorProb(1))/(c3+prm.priorProb(2));

rp(1) = (r1)/(c1);
rp(2) = (r2)/(c2);
rp(3) = (r3)/(c3);
            
pswitch = 0.05;

% choiceIVec  = zeros(3, 8, 3);
% choiceState = zeros(3, 8, 6);
% choicePrb   = zeros(3, 8, 1);
% choiceF     = zeros(3, 8, 1);

for choice = 1 : 3

    choiceOutcome = 1;
    nrm = 0;

    for rew = 0 : 1
        for swtch = 0 : 1
            for subch = 1 : 3

                if swtch == 0 & subch > 1
                    continue;
                end

                fc1 = (c1 + (choice == 1))*(~((swtch==1)&(subch == 1))) + ((swtch==1)&(subch == 1))*2;
                fc2 = (c2 + (choice == 2))*(~((swtch==1)&(subch == 2))) + ((swtch==1)&(subch == 2))*2;
                fc3 = (c3 + (choice == 3))*(~((swtch==1)&(subch == 3))) + ((swtch==1)&(subch == 3))*2;

                fr1 = (r1 + (choice == 1)*rew)*(~((swtch==1)&(subch == 1))) + ((swtch==1)&(subch == 1));
                fr2 = (r2 + (choice == 2)*rew)*(~((swtch==1)&(subch == 2))) + ((swtch==1)&(subch == 2));
                fr3 = (r3 + (choice == 3)*rew)*(~((swtch==1)&(subch == 3))) + ((swtch==1)&(subch == 3)); 
                
                prb = (rew*rp(choice) + (1-rew)*(1-rp(choice)))* ...
                      (swtch*pswitch  + (1-swtch)*(1-pswitch))* ...
                      (swtch*(1/3)    + (1-swtch));
                                      
                kstate = [fr1/fc1 fr2/fc2 fr3/fc3 fc1 fc2 fc3];
                
                qi = isnan(kstate);
                                
                kstate(qi) = 0.5;
                                
                if (fc1 > max(prm.nEval)) | (fc2 > max(prm.nEval)) | (fc3 > max(prm.nEval))
                    
                    if prm.periodic == 1
                        
                        if fc1 > max(prm.nEval)
                            kstate([1 4]) = 0;
                        end
                        
                        if fc2 > max(prm.nEval)
                            kstate([2 5]) = 0;
                        end
                        
                        if fc3 > max(prm.nEval)
                            kstate([3 6]) = 0;
                        end
                    
                        u(choice, choiceOutcome) = prb*f_utility(prm, kstate, betav);
                        
                    else
                    
                        u(choice, choiceOutcome) = 0;
                        
                    end
                    
                else
                    u(choice, choiceOutcome) = prb*f_utility(prm, kstate, betav);
                    if u(choice, choiceOutcome) == 0
%                         fprintf('%d %d %d %d %d %d\n', fc1, fc2, fc3, fr1, fr2, fr3);
                    end
                end

                nrm = nrm + prb;
                
%                 choiceIVec(choice, choiceOutcome, :) = [rew swtch subch];
%                 choicePrb(choice, choiceOutcome, :)  = prb;
%                 choiceF(choice, choiceOutcome, :)    = f_utility(prm, kstate, betav);
%                 choiceState(choice, choiceOutcome, :) = kstate;

                choiceOutcome = choiceOutcome + 1;

            end
        end
    end

    if abs(nrm - 1) > 0.01
        fprintf('bad norm');
    end

end

if prm.aperiodicTransform == 0
    Qsa = rp' + prm.lambda*sum(u, 2);
    Qtran = prm.lambda*sum(u, 2);
else
    Qsa = prm.tau*rp' + prm.tau*prm.lambda*sum(u, 2) + (1-prm.tau)*f_utility(prm, cstate, betav);
    Qtran = prm.tau*prm.lambda*sum(u, 2) + (1-prm.tau)*f_utility(prm, cstate, betav);
end
    
if prm.periodic == 0

    if c1 >= max(prm.nEval)
        Qsa(1)   = 0;
        Qtran(1) = 0;
    end

    if c2 >= max(prm.nEval)
        Qsa(2)   = 0;
        Qtran(2) = 0;
    end

    if c3 >= max(prm.nEval)
        Qsa(3)   = 0;
        Qtran(3) = 0;
    end
    
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [utility_t, Qsi] = stateUtilityNPar(prm, betav)

nEval = length(prm.nEval);
pEval = length(prm.pEval);

Qsi_b     = zeros(pEval, pEval, pEval, nEval, nEval, nEval);
utility_b = zeros(pEval, pEval, pEval, nEval, nEval, nEval);
         
for c1i = 1 : nEval
   
    for c2i = 1 : nEval

        for c3i = 1 : nEval

            for r1i = 1 : pEval
            
                if prm.nEval(c1i) == 0 & prm.pEval(r1i) > 0
                    continue;
                end

                for r2i = 1 : pEval

                    if prm.nEval(c2i) == 0 & prm.pEval(r2i) > 0
                        continue;
                    end

                    for r3i = 1 : pEval

                        if prm.nEval(c3i) == 0 & prm.pEval(r3i) > 0
                            continue;
                        end
                        
                        c1 = prm.nEval(c1i);
                        c2 = prm.nEval(c2i);
                        c3 = prm.nEval(c3i);
                        p1 = prm.pEval(r1i);
                        p2 = prm.pEval(r2i);
                        p3 = prm.pEval(r3i);
                        
                        cstate = [p1 p2 p3 c1 c2 c3];

                        [u, ~] = actionValueNovel(prm, cstate, betav);

                        [mxv, mxi] = max(u);
                        
                        Qsi_b(r1i, r2i, r3i, c1i, c2i, c3i) = mxi;
                        utility_b(r1i, r2i, r3i, c1i, c2i, c3i) = mxv;
                        
                    end
                end
            end
        end
    end    
end

Qsi       = reshape(Qsi_b,(pEval^3)*(nEval^3), 1);
utility_t = reshape(utility_b, (pEval^3)*(nEval^3), 1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [utility_t, Qsi] = stateUtilityParRC(prm, betav)

nEval = length(prm.nEval);
pEval = length(prm.pEval);

Qsi_b     = zeros(nEval, nEval, nEval, nEval, nEval, nEval);
utility_b = zeros(nEval, nEval, nEval, nEval, nEval, nEval);
         
parfor c1i = 1 : nEval
    
    blockMNI = zeros(nEval, nEval, pEval, nEval, nEval);
    blockQsi = zeros(nEval, nEval, pEval, nEval, nEval);

    for c2i = 1 : nEval

        for c3i = 1 : nEval
            
            for r1i = 1 : c1i
                
                for r2i = 1 : c2i
                    
                    for r3i = 1 : c3i
                        
                        c1 = prm.nEval(c1i);
                        c2 = prm.nEval(c2i);
                        c3 = prm.nEval(c3i);
                        p1 = prm.nEval(r1i)/prm.nEval(c1i);
                        p2 = prm.nEval(r2i)/prm.nEval(c2i);
                        p3 = prm.nEval(r3i)/prm.nEval(c3i);
                        
                        cstate = [p1 p2 p3 c1 c2 c3];

                        [u, ~] = actionValueNovel(prm, cstate, betav);

                        [mxv, mxi] = max(u);
                        
                        blockMNI(r1i, r2i, r3i, c2i, c3i) = mxv;
                        blockQsi(r1i, r2i, r3i, c2i, c3i) = mxi;     
                        
                    end
                end
                
            end
        end
    end
    
    Qsi_b(:, :, :, c1i, :, :)    = blockQsi;
    utility_b(:, :, :,c1i, :, :) = blockMNI;
    
end

Qsi       = reshape(Qsi_b,(pEval^3)*(nEval^3), 1);
utility_t = reshape(utility_b, (pEval^3)*(nEval^3), 1);

