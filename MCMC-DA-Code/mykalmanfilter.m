function [xest,logliksum,Pcov] = ...
    mykalmanfilter(T,track,xinit,yall,drawfigure,plotdot,linewidth)
%
% MYKALMANFILTER - Kalman Filter
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
if nargin<5, drawfigure=0; end
if nargin<6, plotdot='b:'; end
if nargin<7, linewidth=2; end

xest = zeros(4,T);
Pcov = zeros(4,4,T);
loglik = zeros(1,T);
if length(track)>T
    track(T+1:length(track)) = 0;
end
vtrack = find(track>0);
if isempty(vtrack),
    logliksum=0;
    return;
end
xest(:,vtrack(1)) = xinit;
Pcov(:,:,vtrack(1)) = eye(4); 

% filtering
for t=2:length(vtrack)
    dT = vtrack(t)-vtrack(t-1);
    Amat = [1 0 dT 0; 0 1 0 dT; 0 0 1 0; 0 0 0 1];
    Gmat = [dT^2/2 0; 0 dT^2/2; dT 0; 0 dT];
    xpred = Amat*xest(:,vtrack(t-1));
    Ppred = Amat*Pcov(:,:,vtrack(t-1))*Amat' + Gmat*G.Qcov*Gmat';
    innov = (yall{vtrack(t)}(track(vtrack(t)),:)' - G.Cmat*xpred);
    Bmat = G.Cmat*Ppred*G.Cmat' + G.Rcov;
    invBmat = inv(Bmat);
    K_1 = Ppred*G.Cmat'*invBmat;
    xest(:,vtrack(t)) = xpred + K_1*innov;
    Pcov(:,:,vtrack(t)) = Ppred - K_1*G.Cmat*Ppred;
    loglik(vtrack(t)) = mvnormpdf(innov,zeros(2,1),Bmat,1);
    if drawfigure 
        plot([xest(1,vtrack(t-1)) xest(1,vtrack(t))]',...
             [xest(2,vtrack(t-1)) xest(2,vtrack(t))]',plotdot,'LineWidth',linewidth);
    end
end
logliksum = sum(loglik(:));
