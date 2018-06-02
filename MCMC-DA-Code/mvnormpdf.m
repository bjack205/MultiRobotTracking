function pd = mvnormpdf(X,MU,K,dolog)
%
% MVNORMPDF(X,MU,K,dolog)
%   - calculates the density at X of the mutivariate normal 
%     distribution. I.e. N(X | MU,K)
%   - inputs X and MU are expected to be column/row vectors
%   - K is the covariance matrix (nonsingular)
%
%   Copyright (c) 2003-2004, 2010 Songhwai Oh
%       CPSLAB, SNU (http://cpslab.snu.ac.kr)
%
if nargin<4, dolog=0; end

[N,M] = size(X);
if M>1,
    X = X';
    MU = MU';
    N = M;
end
alpha = (2*pi)^(N/2)*sqrt(abs(det(K)));
beta = sum(((X-MU)'*inv(K)).*(X-MU)',2);   % Chris Bregler's trick (from BNT)
if dolog
    pd = -.5*beta - log(alpha); 
else
    pd = exp(-.5*beta)/alpha;
    %pd = exp(-.5*tr((X-MU)'*inv(K)*(X-MU)))/alpha;
end
