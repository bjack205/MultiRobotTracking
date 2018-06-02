%
% DEMO - Demo of Multi-scan MCMCDA 
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

% gen_trajectory(trajname,ntracks,tmax,tlen,L,vmax,vmin,randombirth)
gen_trajectory('demo_traj',10,50,30,100,5,1,1); 

inval = input('continue (y/n)? ','s');
if strcmp(inval,'n') || strcmp(inval,'N')
    return
end

% gen_scenarios(sname,tname,nfa,dpr,doplot,stepon)
gen_scenarios('demo_scen','demo_traj',10,.9,1,0);

inval = input('continue (y/n)? ','s');
if strcmp(inval,'n') || strcmp(inval,'N')
    return
end

% scen_tracking(scenfile,savefile,greedyinit,nmcmc,winsize,winsize0,minlen,showfig,verbose)
scen_tracking('demo_scen','demo_mcmcda',0,2000,20,10,5,1,0);