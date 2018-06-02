function plot_results(scenfile,savefile)
%
% PLOT_RESULTS 
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
if nargin<3, scenonly=0; end
if nargin<4, stepon=0; end
if nargin<5, saveplot=0; end

scen = load(scenfile);
result = load(savefile);

G = scen.gvs;
Tall = scen.gvs.TMAX; 

figure
axes('Box','on');
axis([scen.gvs.SR(1,1),scen.gvs.SR(1,2),scen.gvs.SR(2,1),scen.gvs.SR(2,2)]);
axis square
title('Tracking results');
xlabel('x'); ylabel('y');
hold on;

    lineshape = 'r-';
    linewidth=2;

ally = [];
for ti=1:Tall
    ally = [ally; scen.ygold{ti}];
end
if ~scenonly
    %plot(ally(:,1),ally(:,2),'k.');
end
    
if ~scenonly
    for k=1:size(result.gvs.record.fulltrackinfo.track,2)
        [dummy1,dummy2] = ...
            mykalmanfilter(Tall,result.gvs.record.fulltrackinfo.track(:,k),...
            result.gvs.record.fulltrackinfo.xinit(:,k),scen.ygold,1,'b-',linewidth);
    end
end

drawnow
