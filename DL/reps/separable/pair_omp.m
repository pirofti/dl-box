% Copyright (c) 2016 Paul Irofti <paul@irofti.net>
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

function [X,shared] = pair_omp(Y,D,s,shared,varargin)
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

    fun_pair_omp = @pair_omp_s;
    error = 0;
    if ~isempty(varargin) && strcmp(varargin{1}, 'error')
        fun_pair_omp = @pair_omp_err;
        error = varargin{2};
        if length(varargin) > 2
            s = varargin{3};
        else
            s = size(Y,1)/2;
        end
    end
    N = size(Y,3);
    X = zeros(size(D{1},2),size(D{2},1),N);
    for k = 1:N
        X(:,:,k) = fun_pair_omp(Y(:,:,k),D{1},D{2},s,error);
    end
end