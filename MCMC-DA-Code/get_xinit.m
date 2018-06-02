function xinit = get_xinit(track,yall)
%
% GET_XINIT - Get initial states of a track
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

vtrack = find(track>0);
xinit = zeros(4,1);
if length(vtrack)>1
    d = vtrack(2)-vtrack(1);
    xdelta = [yall{vtrack(2)}(track(vtrack(2)),1)-yall{vtrack(1)}(track(vtrack(1)),1),...
              yall{vtrack(2)}(track(vtrack(2)),2)-yall{vtrack(1)}(track(vtrack(1)),2)]./(d*G.F_T);
    xinit = [yall{vtrack(1)}(track(vtrack(1)),1),yall{vtrack(1)}(track(vtrack(1)),2),...
            xdelta(1),xdelta(2)]';
else
    xinit = [yall{vtrack(1)}(track(vtrack(1)),1),yall{vtrack(1)}(track(vtrack(1)),2),0,0]';
end
