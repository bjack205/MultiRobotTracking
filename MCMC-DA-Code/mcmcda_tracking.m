function toptrack = mcmcda_tracking(Tnow,do_toptracking) 
%
% MCMCDA_TRACKING - Second-level wrapper for multi-scan MCMCDA
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

global gvs;
if nargin<2, do_toptracking = 0; end

Tinit = max(1,Tnow-gvs.winsize+1);
gvs.yobs_win = gvs.yobs(Tinit:Tnow);
Tf = Tnow - Tinit + 1;
if Tnow>gvs.winsize
    Ts = ceil(Tf/2);
else
    Ts = 1;
end
nmcmc_now = gvs.nmcmc;
depth_now = min(gvs.depth,Tf);
minlen_now = min(gvs.minlen,Tf);

toptrack = [];
if Tnow>1
    if isempty(gvs.record.trackinfo.track)
        gvs.record.trackinfo = mcmcda(gvs.mcmcda,gvs.yobs_win,Ts,Tf,nmcmc_now, ...
            depth_now,minlen_now,gvs.greedyinit,gvs.delta);
    else
        pt_siz = size(gvs.record.trackinfo.track);
        track0 = zeros(Tf,pt_siz(2));
        if Tinit<=gvs.record.trackinfo.times(2)
            pwin_ti = Tinit - gvs.record.trackinfo.times(1) + 1;
            pwin_len = pt_siz(1) - pwin_ti + 1;
            track0(1:pwin_len,:) = gvs.record.trackinfo.track(pwin_ti:pt_siz(1),:);
        end
        
        if 1        
            % predict
            pred.nt = size(gvs.yobs_win{Tf},1);
            pred.num_tracks = size(track0,2);
            pred.dist = inf*ones(pred.num_tracks,pred.nt);
            pred.vT = [];
            pred.vY = [];
            for k=1:pred.num_tracks
                vtrack = find(track0(:,k)>0);
                if (vtrack(end)==Tf-1) 
                    X0 = get_xinit(track0(:,k),gvs.yobs_win);
                    [X,dummy,P] = ...
                        mykalmanfilter(Tf,track0(:,k),X0,gvs.yobs_win,0);
                    xpred = gvs.mcmcda.Amat*X(:,Tf-1);
                    Ppred = gvs.mcmcda.Amat*P(:,:,Tf-1)*gvs.mcmcda.Amat' ...
                        + gvs.mcmcda.Gmat*gvs.mcmcda.Qcov*gvs.mcmcda.Gmat';
                    Bmat = gvs.mcmcda.Cmat*Ppred*gvs.mcmcda.Cmat' + gvs.mcmcda.Rcov;
                    for n=1:pred.nt
                        innov = (gvs.yobs_win{Tf}(n,:)' - gvs.mcmcda.Cmat*xpred);
                        if norm(gvs.yobs_win{Tf}(n,:)-...
                                gvs.yobs_win{Tf-1}(track0(Tf-1,k),:))<gvs.mcmcda.ValR
                            pred.dist(k,n) = innov'*inv(Bmat)*innov;
                            pred.vT = union(pred.vT,k);
                            pred.vY = union(pred.vY,n);
                        end
                    end
                end
            end
            [assign,C] = assignmentoptimal(pred.dist(pred.vT,pred.vY));
            for k=1:length(pred.vT)
                if assign(k)>0
                    track0(Tf,pred.vT(k)) = pred.vY(assign(k));
                end
            end
        end
        
        gvs.record.trackinfo = mcmcda(gvs.mcmcda,gvs.yobs_win,Ts,Tf,nmcmc_now, ...
            depth_now,minlen_now,0,gvs.delta,track0);
    end
    gvs.record.trackinfo.times = [Tinit,Tnow];
    num_tracks = size(gvs.record.trackinfo.track,2);
    if do_toptracking && num_tracks>0
        toptrack.pos = cell(1,num_tracks);
        toptrack.loglik = zeros(1,num_tracks);
        toptrack.cov = cell(1,num_tracks);
        for k=1:num_tracks
            [toptrack.pos{k},toptrack.loglik(k),toptrack.cov{k}] = ...
                mykalmanfilter(size(gvs.yobs_win,2),gvs.record.trackinfo.track(:,k),...
                gvs.record.trackinfo.xinit(:,k),gvs.yobs_win,0);
        end
    end
end 

