function plottraj(M,T,xtimes,xtrajs,yall,stepon,notraj,nonewfigure,noclear)
%
% PLOTTRAJ
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
if nargin<6, stepon=0; end
if nargin<7, notraj=0; end
if nargin<8, nonewfigure=0; end
if nargin<9, noclear=0; end

ccp = 'r.';
ccl = 'r.-';
csp = 'bo';
csl = 'b:';
cob = 'k.';

if nonewfigure
    if ~noclear, clf; end
else
    figure;
    axes('Box','on');
    axis square
    xlabel('x'); ylabel('y');
end
hold on;
if isvarname('G')
    %axis([G.SR(1,1),G.SR(1,2),G.SR(2,1),G.SR(2,2)]);
    axis square
end
for t=1:T
    if ~notraj
        for m=1:M
            if t==xtimes(m,1)
                plot(xtrajs{m}(1,1),xtrajs{m}(2,1),ccp);
            elseif t>xtimes(m,1) && t<=xtimes(m,2)
                plot([xtrajs{m}(1,t-xtimes(m,1)),xtrajs{m}(1,t-xtimes(m,1)+1)],...
                    [xtrajs{m}(2,t-xtimes(m,1)),xtrajs{m}(2,t-xtimes(m,1)+1)],ccl,'LineWidth',2);
            end
        end
    end
    if ~nonewfigure 
        yt = yall{t};
        ny = size(yt,1);
        for n=1:ny
            plot(yt(n,1),yt(n,2),cob,'MarkerSize',10);
        end
    end
    
    if stepon, 
        pause; 
        clf; axis([G.SR(1,1),G.SR(1,2),G.SR(2,1),G.SR(2,2)]); axis square
        hold on; 
    end
end
