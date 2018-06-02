function [xkmat] = gen_trajectory(trajname,ntracks,tmax,tlen,L,vmax,vmin,randombirth)
%
% GEN_TRAJECTORY - Creates a set of trajectories of multiple targets
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

G.M = ntracks;                     % max number of targets
G.TMAX = tmax;                     % number of time steps
G.MIN_TRACK_LENGTH = tlen;
G.DIM = 4;                         % dimension of state space for targets
G.L = L;
G.SR = [0 G.L; 0 G.L];             % surveillance region
G.SRV = prod(G.SR(:,2)-G.SR(:,1)); % volume of surveillance region
G.NTR = ntracks/G.SRV/G.TMAX;
G.TTR = .01;
G.BIRTH = G.TMAX;
G.LIFEXP = G.TMAX;

G.F_VEL_MAX   = vmax;
G.F_VEL_MIN   = vmin;
G.F_VEL_INIT  = vmin + .5*(vmax-vmin);

G.F_T = 1;
dT = G.F_T;
G.Amat = [1 0 dT 0; 0 1 0 dT; 0 0 1 0; 0 0 0 1];
G.Gmat = [dT^2/2 0; 0 dT^2/2; dT 0; 0 dT];
G.Cmat = [1 0 0 0; 0 1 0 0];
G.Qsgm = G.F_VEL_INIT/10 * [1;1];
G.Qcov = diag(G.Qsgm.^2);
G.Rsgm = G.F_VEL_INIT/10 * [1;1];
G.Rcov = diag(G.Rsgm.^2);
G.ValR = G.F_T*G.F_VEL_MAX;

xtimes = zeros(G.M,2);
xtrajs = cell(G.M,1);
xidmat = zeros(G.M,G.TMAX);

numtracks = 1;
while numtracks<=G.M
    x = [];
    if randombirth
        btime = unimultrnd(G.BIRTH);
        if btime>=G.TMAX
            continue;
        end
        dtime = min(G.TMAX, btime + floor(exprnd(G.LIFEXP)));
        x = zeros(G.DIM,dtime-btime+1);
        scale = min(1,G.F_VEL_INIT*G.TMAX/G.L);
        pos = scale*G.L*rand(2,1) + (1-scale)/2*G.L*ones(2,1); 
        theta = 2*pi*rand(1);
        vel = [G.F_VEL_INIT*cos(theta);G.F_VEL_INIT*sin(theta)];
        x(:,1) = [pos;vel];
    else
        btime = min(G.TMAX-G.MIN_TRACK_LENGTH, 1+(numtracks-1)*(G.TMAX/G.MIN_TRACK_LENGTH));
        dtime = min(G.TMAX, btime + G.MIN_TRACK_LENGTH);
        x = zeros(G.DIM,dtime-btime+1);
        scale = min(1,G.F_VEL_INIT*G.TMAX/G.L);
        pos = scale*G.L*rand(2,1) + (1-scale)/2*G.L*ones(2,1); 
        theta = 2*pi*rand(1);
        vel = [G.F_VEL_INIT*cos(theta);G.F_VEL_INIT*sin(theta)];
        x(:,1) = [pos;vel];
    end
    
    for t=2:dtime-btime+1
        xrnd = G.Gmat * (G.Qsgm.*randn(2,1));
        x(:,t) = G.Amat*x(:,t-1) + xrnd;
        if any(x(1:2,t)>G.SR(:,2)) || any(x(1:2,t)<G.SR(:,1)) ...
                || norm(x(3:4,t))>G.F_VEL_MAX || norm(x(3:4,t))<G.F_VEL_MIN
            x(:,t) = zeros(G.DIM,1);
            dtime = t-1;
            break
        end
    end

    if btime>0 && dtime-btime+1>=G.MIN_TRACK_LENGTH
        xtimes(numtracks,:) = [btime,dtime];
        xidmat(numtracks,btime:dtime) = 1;
        xtrajs{numtracks} = x(:,1:dtime-btime+1);
        numtracks = numtracks + 1;
        fprintf(1,'.');
    end
end
xkmat = sum(xidmat,1);

fprintf(' [ ');
for t=1:G.TMAX
    fprintf('%d ',sum(xidmat(:,t)>0));
end
fprintf(']\n');

gvs = G;
save(trajname,'gvs','xtrajs','xtimes','xidmat','xkmat');

figure;
axes('Box','on');
axis([gvs.SR(1,1),gvs.SR(1,2),gvs.SR(2,1),gvs.SR(2,2)]);
axis square
title('Trajectories');
xlabel('x'); ylabel('y');
plottraj(G.M,G.TMAX,xtimes,xtrajs,cell(1,G.TMAX),0,0,1,1);

