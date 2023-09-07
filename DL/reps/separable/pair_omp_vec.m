% Copyright (c) 2020 Paul Irofti <paul@irofti.net>
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

function [X,shared] = pair_omp_vec(Y,D,s,shared,varargin)
%% 2D Orthogonal Matching Pursuit algorithm (for separable dictionaries)
% INPUTS:
%   Y -- training signals set
%   D -- current dictionary
%   s -- sparsity constraint
% PARAMETERS:
%   error -- switch to error driven OMP
%   max_s -- maximum allowed density for error OMP
%
% OUTPUTS:
%   X -- sparse representations
    [p1, p2, N] = size(Y);
    n1 = size(D{1}, 2);
    n2 = size(D{2}, 1);
    Yv = reshape(Y, p1*p2, N);
    Dv = kron(D{2}',D{1});
    Xv = omp(Yv, Dv, s, shared, varargin{:});
    X = reshape(Xv, n1, n2, N);
end