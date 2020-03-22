% Copyright (c) 2017-2018 Bogdan Dumitrescu <bogdan.dumitrescu@acse.pub.ro>
% Copyright (c) 2019 Andra Băltoiu <andra.baltoiu@fmi.unibuc.ro>
% 
% Permission to use, copy, modify, and/or distribute this software for any
% purpose with or without fee is hereby granted, provided that the above
% copyright notice and this permission notice appear in all copies.
% 
% THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
% WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
% MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
% ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
% WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
% ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
% OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

function [D, R] = rls_dl(Y, D, s, iternum, ff, R)

% RLS DL (Skretting & Engan, 2010)
% Input:
%   Y        - signal matrix
%   D        - initial dictionary
%   s        - desired sparsity level
%   iternum  - number of DL iterations
%   ff       - forgetting factor
% Output:
%   D        - learned dictionary
%   X        - final representation matrix
%   err      - RMSE values for all iterations

% BD 26.05.2017


% prepare
[m,n] = size(D);
[~,N] = size(Y);
ff = 1/ff;          % we need only the inverse of the forgetting factor

opts1.UT = true;
opts1.TRANSA = true;
opts2.UT = true;


for t = 1 : iternum
    y = Y(:,rem(t-1,N)+1);       % artificially extract current signal
    x = omp(y,D,s);              % compute current representation

    r = y - D*x;                 % residual


    u = linsolve(R, x, opts1);
    u = ff * linsolve(R, u, opts2);
    a = 1 / (1 + x'*u);
    R = cholupdate(R,x);         % update Cholesky factorization


    D = D + a*r*u';
    D = normc(D);

end
