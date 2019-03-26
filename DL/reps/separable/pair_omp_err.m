% Copyright (c) 2017 Paul Irofti <paul@irofti.net>
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

function [X] = pair_omp_err(Y, D1, D2, max_s, err)
%% 2D OMP with error stopping criteria
% INPUTS:
%   Y -- dense patch
%   D1 -- left dictionary
%   D2 -- right dictionary
%   max_s -- maximum allowed density
%   err -- error threshold
%
% OUTPUTS:
%   X -- sparse representation
    I = []; J = [];
    R = Y;
    n1 = size(D1,2);
    n2 = size(D2,1);
    k = 1;
    while norm(R,'fro') > err && k <= max_s
        P = D1'*R*D2';
        [~, ind] = max(abs(P(:)));
        [I(k), J(k)] = ind2sub([n1, n2],ind);
        % The naive way
        %{
        A = [];
        for i = 1:k
            A = [A kron(D2(J(i),:)',D1(:,I(i)))];
        end
        %}
        % The smart way from SO
        % http://stackoverflow.com/a/41873232/1565442
        A = reshape(permute(bsxfun(@times, D2(J,:).', permute(D1(:,I), [3 2 1])), [3 1 2]), [], k);
        x = pinv(A)*Y(:);
        R = Y - D1(:,I)*diag(x)*D2(J,:);
        k = k + 1;
    end
    if k == 1
        X = zeros(n1,n2);  % norm(Y) < err, yes it can happen..
    else
        X = full(sparse(I,J,x,size(D1,2), size(D2,1)));
    end
end