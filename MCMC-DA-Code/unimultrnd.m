function sample = unimultrnd(N)
%
% UNIMULTRND(N) Generates a uniform multinomial random number
%   N - number of possible values
%
%   Copyright (c) 2003-2004, 2010 Songhwai Oh
%       CPSLAB, SNU (http://cpslab.snu.ac.kr)
%
rnd = rand(1);
for n=1:N
    if rnd < n/N
        sample = n;
        break;
    end
end