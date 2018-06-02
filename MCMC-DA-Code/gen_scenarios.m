function [xkmat,ny] = gen_scenarios(sname,tname,nfa,dpr,doplot,stepon)
%
% GEN_SCENARIOS - Creates a senario based on trajectories from 'tname'
%
%   Reference:
%
%   [1] Songhwai Oh, Stuart Russell, and Shankar Sastry, "Markov Chain Monte
%       Carlo Data Association for Multi-Target Tracking," IEEE Transactions 
%       on Automatic Control, vol. 54, no. 3, pp. 481-497, Mar. 2009.
%
%   Copyright (c) 2003-2004, 2010 Songhwai Oh
%       CPSLAB, SNU (http://cpslab.snu.ac.kr)
%

global G

load(tname);

G = gvs;

G.DP = dpr;
G.FAR = nfa/G.SRV;

ygold = cell(G.TMAX,1);
scentrack.track = zeros(G.TMAX,G.M);
for t=1:G.TMAX
    y_t = [];
    yfromx = zeros(G.M,2);
    for m=1:G.M
        if xidmat(m,t)==1
            if rand(1) < G.DP
                y_tj = G.Cmat*xtrajs{m}(:,t-xtimes(m,1)+1) + (G.Rsgm.*randn(2,1));
                y_t = [y_t; y_tj'];
                yfromx(m,:) = y_tj';
            end
        end
    end
    % false alarms
    nfa = poissrnd(G.FAR*G.SRV);
    for n=1:nfa
        y_tj = (rand(2,1).*(G.SR(:,2)-G.SR(:,1)) + G.SR(:,1))';
        y_t = [y_t; y_tj];
    end
    ygold{t} = y_t(randperm(size(y_t,1)),:);
    for m=1:G.M
        if all(yfromx(m,:)>0)
            for n=1:size(ygold{t},1)
                if all(ygold{t}(n,:)==yfromx(m,:))
                    scentrack.track(t,m)=n;
                end
            end
        end
    end
end
ny = zeros(1,G.TMAX);
for t=1:G.TMAX
    ny(t) = size(ygold{t},1);
end

scentrack.xinit = zeros(4,G.M);
scentrack.loglik = zeros(1,G.M);
scentrack.xest = zeros(4,G.TMAX,G.M);
for k=1:G.M
    scentrack.xinit(:,k) = get_xinit(scentrack.track(:,k),ygold);
end
for k=1:G.M
    [scentrack.xest(:,:,k),scentrack.loglik(k)] = ...
        mykalmanfilter(G.TMAX,scentrack.track(:,k),scentrack.xinit(:,k),ygold,0);
end
scentrack.fullloglik = trackloglik(G.TMAX,scentrack.track,scentrack.loglik,ny);

gvs = G;
save(sname,'gvs','xtimes','xidmat','xtrajs','ygold','scentrack');
if doplot
    plottraj(G.M,G.TMAX,xtimes,xtrajs,ygold,stepon);
end
