function F1 = scen_tracking(scenfile,savefile,greedyinit,nmcmc,winsize,winsize0,minlen,showfig,verbose)
%
% SCEN_TRACKING - Top-level wrapper for multi-scan MCMCDA
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

global gvs G

scen = load(scenfile);

Tall = scen.gvs.TMAX; 
gvs.scenfile = scenfile;

% global variables for mcmc data association
gvs.mcmcda.ValR = scen.gvs.ValR;    % max velocity x sampling period
gvs.mcmcda.NTR = scen.gvs.NTR;      % new target rate
gvs.mcmcda.TTR = scen.gvs.TTR;      % target termination rate
gvs.mcmcda.SRV = scen.gvs.SRV;      % surveillance region volume
gvs.mcmcda.F_T = scen.gvs.F_T;      % sampling period
gvs.mcmcda.Amat = scen.gvs.Amat;    % dynamic & observation model
gvs.mcmcda.Cmat = scen.gvs.Cmat;    % dynamic & observation model
gvs.mcmcda.Gmat = scen.gvs.Gmat;    % dynamic & observation model
gvs.mcmcda.Qcov = scen.gvs.Qcov;    % process covariance
gvs.mcmcda.Qsgm = scen.gvs.Qsgm;    % sqrt of process covariance
gvs.mcmcda.Rsgm = scen.gvs.Rsgm;    % sqrt of observation covariance
gvs.mcmcda.Rcov = scen.gvs.Rcov;    % observation covariance
gvs.mcmcda.FAR = scen.gvs.FAR;      % false alarm rate
gvs.mcmcda.DP = scen.gvs.DP;        % detection probability
gvs.mcmcda.F_VEL_MAX = scen.gvs.F_VEL_MAX;

G = gvs.mcmcda;

% global variables for 'tracking'
gvs.T = scen.gvs.F_T;   % sampling period
gvs.winsize = winsize;  % observation window size
gvs.winsize0 = winsize0;
gvs.nmcmc = nmcmc;      
gvs.depth = 5;
gvs.minlen = minlen;
gvs.greedyinit = greedyinit;
gvs.delta = .5;
gvs.yobs_win = []; % current observation window
gvs.record.trackinfo = [];    % previos track information
gvs.record.trackinfo.track = [];
gvs.record.trackinfo.times = [0,0];
gvs.record.fulltrackinfo = [];
gvs.record.fulltrackinfo.track = [];

gvs.record.runtime = cputime;
gvs.record.cputime = zeros(1,Tall);

for t=1:Tall
    
    % get observations
    gvs.yobs = scen.ygold(1:t);
    
    % tracking ... 
    gvs.record.cputime(t) = cputime;
    mcmcda_tracking(t,1);
    gvs.record.cputime(t) = cputime - gvs.record.cputime(t);
    
    %%% FULL TRACK
    if gvs.winsize0==Tall
        gvs.record.fulltrackinfo.track = gvs.record.trackinfo.track;
    else
        Tnow = t;
        Tinit = max(1,Tnow-gvs.winsize+1);
        Tf = Tnow - Tinit + 1;
        if Tnow>gvs.winsize
            full_T = ceil(Tf/2);
            part_uT = ceil(Tf/2);
        else
            full_T = 1;
            part_uT = 1;
        end
        full_K = 0;
        if Tnow==gvs.winsize
            if ~isempty(gvs.record.trackinfo.track)
                [part_T,part_K] = size(gvs.record.trackinfo.track);
                full_T = ceil(gvs.winsize/2);
                kf = 0;
                for kp=1:part_K
                    if any(gvs.record.trackinfo.track(1:full_T,kp)~=0)
                        kf = kf + 1;
                        gvs.record.fulltrackinfo.track(1:full_T,kf) = ...
                            gvs.record.trackinfo.track(1:full_T,kp);
                    end
                end
            end
        elseif Tnow>gvs.winsize
            full_T = Tinit + ceil(Tf/2) - 2; 
            if ~isempty(gvs.record.fulltrackinfo.track)
                [full_T,full_K] = size(gvs.record.fulltrackinfo.track);
                gvs.record.fulltrackinfo.track(full_T+1,1:full_K) = 0;
            end
            [part_T,part_K] = size(gvs.record.trackinfo.track);
            for kf=1:full_K
                for kp=1:part_K
                    vtrack = find(gvs.record.fulltrackinfo.track(:,kf)>0);
                    full_tf = vtrack(length(vtrack));
                    if full_tf>=Tinit
                        part_tf = full_tf-Tinit+1;
                        if gvs.record.fulltrackinfo.track(full_tf,kf) ...
                                == gvs.record.trackinfo.track(part_tf,kp)
                            if Tnow < Tall
                                gvs.record.fulltrackinfo.track(full_tf:full_T+1,kf) = ...
                                    gvs.record.trackinfo.track(part_tf:part_uT,kp);
                            else
                                gvs.record.fulltrackinfo.track(full_tf:Tall,kf) = ...
                                    gvs.record.trackinfo.track(part_tf:part_T,kp);
                            end
                        end
                    end
                end
            end
            kf = full_K;
            for kp=1:part_K
                % add new tracks
                if all(gvs.record.trackinfo.track(1:part_uT-1,kp)==0) ...
                        && gvs.record.trackinfo.track(part_uT,kp)>0
                    kf = kf + 1;
                    if Tnow < Tall
                        gvs.record.fulltrackinfo.track(full_T+1,kf) = ...
                            gvs.record.trackinfo.track(part_uT,kp);
                    else
                        gvs.record.fulltrackinfo.track(full_T+1:Tall,kf) = ...
                            gvs.record.trackinfo.track(part_uT:part_T,kp);
                    end
                end
            end
            
        end
        if Tnow==Tall && ~isempty(gvs.record.fulltrackinfo.track)
            full_T = size(gvs.record.fulltrackinfo.track,1);
            if full_T<Tall
                gvs.record.fulltrackinfo.track(full_T+1:Tall,:) = 0;
            end
        end
    end
    
    % plot
    if showfig
        if t==1, figure;
        else    clf;
        end
        axis([scen.gvs.SR(1,1),scen.gvs.SR(1,2),scen.gvs.SR(2,1),scen.gvs.SR(2,2)]);
        axis square
        hold on
        if ~isempty(gvs.yobs{t})
            plot(gvs.yobs{t}(:,1),gvs.yobs{t}(:,2),'k.');
        end
        if ~isempty(gvs.record.trackinfo.track)
            for k=1:size(gvs.record.trackinfo.track,2)
                [dummy1,dummy2] = ...
                    mykalmanfilter(length(gvs.yobs_win),gvs.record.trackinfo.track(:,k),...
                    gvs.record.trackinfo.xinit(:,k),gvs.yobs_win,1,'b:',2);
            end
        end
        if ~isempty(gvs.record.fulltrackinfo.track)
            [full_T,full_K] = size(gvs.record.fulltrackinfo.track);
            fullxinit = zeros(4,full_K);
            for k=1:full_K
                fullxinit(:,k) = get_xinit(gvs.record.fulltrackinfo.track(:,k),gvs.yobs(1:full_T));
                [dummy1,dymmy2,dummy3] ...
                    = mykalmanfilter(full_T,gvs.record.fulltrackinfo.track(:,k),...
                    fullxinit(:,k),gvs.yobs(1:full_T),1,'r-',2);
            end
        end
        drawnow
    end
    if verbose
        fprintf('[%d] K=%d time=%.02fs\n', t, size(gvs.record.trackinfo.track,2),gvs.record.cputime(t));
    end
end

full_K = size(gvs.record.fulltrackinfo.track,2);
gvs.record.fulltrackinfo.xinit = zeros(4,full_K);
gvs.record.fulltrackinfo.loglik = zeros(1,full_K);
gvs.record.fulltrackinfo.xest = zeros(4,Tall,full_K);
gvs.record.fulltrackinfo.Pcov = zeros(4,4,Tall,full_K);

for k=1:full_K
    gvs.record.fulltrackinfo.xinit(:,k) = ...
        get_xinit(gvs.record.fulltrackinfo.track(:,k),gvs.yobs);
    [gvs.record.fulltrackinfo.xest(:,:,k),...
            gvs.record.fulltrackinfo.loglik(k),...
            gvs.record.fulltrackinfo.Pcov(:,:,:,k)] ...
        = mykalmanfilter(Tall,gvs.record.fulltrackinfo.track(:,k),...
        gvs.record.fulltrackinfo.xinit(:,k),gvs.yobs,0);
    % interpolate missing parts
    vtrack = find(gvs.record.fulltrackinfo.track(:,k)>0);
    vtrack_len = length(vtrack);
    for vt=2:vtrack_len
        d = vtrack(vt) - vtrack(vt-1); 
        if d>1
            x1 = gvs.record.fulltrackinfo.xest(1:2,vtrack(vt),k);
            x0 = gvs.record.fulltrackinfo.xest(1:2,vtrack(vt-1),k);
            for dt=1:d-1
                gvs.record.fulltrackinfo.xest(1:2,vtrack(vt-1)+dt,k) = x0 + (x1 - x0).*(dt/d);
            end
        end
    end
end
gvs.record.runtime = cputime - gvs.record.runtime;

ny = zeros(1,Tall);
for t=1:Tall
    ny(t) = size(gvs.yobs{t},1);
end
gvs.record.fulltrackinfo.fullloglik = ...
    trackloglik(Tall,gvs.record.fulltrackinfo.track,gvs.record.fulltrackinfo.loglik,ny);

save(savefile,'gvs');

if showfig
    plot_results(scenfile,savefile);
end
