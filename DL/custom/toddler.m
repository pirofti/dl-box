% Copyright (c) 2018 Bogdan Dumitrescu <bogdan.dumitrescu@acse.pub.ro>
% Copyright (c) 2018, 2019 Paul Irofti <paul@irofti.net>
% Copyright (c) 2019 Andra Baltoiu <andra.baltoiu@fmi.unibuc.ro>
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

% Cite as:
% P. Irofti and A. Băltoiu, Malware Identification with Dictionary Learning,
% in 27th European Signal Processing Conference (Eusipco), 
% 2019 IEEE International Conference on, 2019, pp. 1--5.

function estimate = toddler(Y, D, W, A, varargin)

% INPUTS:
%   Y                       signals, (m x N) matrix
%   D                       pre-trained dictionary
%   W                       pre-trained classifier matrix
%   A                       pre-trained label consistency transformation
%   Optional:
%       Sparsity
%       Constraint          model parameters {alpha, beta, lambda1, lambda2}
%                           - Label Consistent (LC-KSVD) alpha and beta parameters
%                           - (FIXED) Tikhonov parameters lambda1 and lambda2
%                             for regularizing the classifier (W) and
%                             discriminative dictionary (A) respectively.
%       Forget              controls the contribution of past signals
%       Method              update methods {'fixed', 'G2', 'WA2'}. Default: 'fixed'
%                           'fixed' - use fixed-valued Tikhonov parameters for
%                                     updating W and A 
%                           'G2'    - use the 2-norm of G as Tikhonov parameter for
%                                     updating both W and A
%                           'WA2'   - use the 2-norm of W and A respectively as 
%                                     Tikhonov parameters to independently update W and A
%
% OUTPUT: estimate          estimated class label


% Arguments sanity
if rem(nargin,2), error('Invalid arguments'); end



% CONFIGURATION

[m, N] = size(Y);
n = size(D,2);              % number of dictionary atoms

% Defaults

PARAMS.Classes = 2;
PARAMS.Sparsity = sqrt(n);
PARAMS.Constraint = {4,16,16,16};
PARAMS.Forget = 0.999;
PARAMS.Method = 'fixed';
PARAMS.Labels = [];

% Set parameter values

for i = 1: 2: nargin - 4
    param = varargin{i};
    value = varargin{i+1};
    switch param
        case 'Classes',    PARAMS.Classes = value;
        case 'Sparsity',   PARAMS.Sparsity = value;
        case 'Constraint', PARAMS.Constraint = value;
        case 'Forget',     PARAMS.Forget = value;
        case 'Method',     PARAMS.Method = value;
        case 'Labels',     PARAMS.Labels = value;
        otherwise, error('Undefined parameter %s \n', param);
    end
end



% INITIALIZATION

nc = floor(n/(PARAMS.Classes + 1)); % atoms per class
nr = nc*(PARAMS.Classes + 1);
me = m + PARAMS.Classes + nr;       % total dictionary size
w = zeros(PARAMS.Classes,N);
estimate = zeros(N,1);
Nl = length(PARAMS.Labels);         % number of labeled samples



% TODDLER

G = eye(n);
for i = 1:N
    y = Y(:,i);
    x = omp(y,D,PARAMS.Sparsity);            % sparse representation
    if i > Nl
        % Classification
        w(:,i) = W*x/norm(W*x);
        [~, I] = maxk(abs(w(:,i)),2);
    else
        I(1) = PARAMS.Labels(i);                 % known label
    end
    
    h = zeros(PARAMS.Classes,1);                 % label
    h(I(1)) = 1;
    q = zeros(nr,1);                             % label consistency
    q((I(1)-1)*nc+1:I(1)*nc) = 1;
    
    q(PARAMS.Classes*nc+1 : end) = 1;            % shared dict
    
    
    % Pack
    ye = [y; sqrt(PARAMS.Constraint{1})*h; sqrt(PARAMS.Constraint{2})*q];
    De(1:m,:) = D;
    De(m+1 : m+PARAMS.Classes, :) = sqrt(PARAMS.Constraint{1}) * W;
    De(m+PARAMS.Classes+1 : me, :) = sqrt(PARAMS.Constraint{2}) * A;
    De = normc(De);
    
    [De, G] = rls_dl(ye, De, PARAMS.Sparsity, 1, PARAMS.Forget, G);
    
    % Unpack
    D = De(1:m,:);
    normD = sqrt(sum(D.*D)+eps);
    D = normc(D);
    
    W = De(m+1:m+PARAMS.Classes, :) / sqrt(PARAMS.Constraint{1});
    W = W ./ repmat(normD, PARAMS.Classes, 1);
    A = De(m+PARAMS.Classes+1 : me, :) / sqrt(PARAMS.Constraint{2});
    A = A ./ repmat(normD, nr, 1);
    
    switch PARAMS.Method
        
        case 'fixed'
            W = (h*x' + PARAMS.Constraint{3}*W)/(x*x' + PARAMS.Constraint{3}*eye(nr));
            A = (q*x' + PARAMS.Constraint{4}*A)/(x*x' + PARAMS.Constraint{4}*eye(nr));
            
        case 'G2'
            lambda = svds(G,1);
            W = (h*x' + lambda*W)/(x*x' + lambda*eye(nr));
            A = (q*x' + lambda*A)/(x*x' + lambda*eye(nr));
            
        case 'WA2'
            W = De(m+1:m + PARAMS.Classes, :) / sqrt(PARAMS.Constraint{1});
            W = W ./ repmat(normD, res.c, 1);
            A = De(m+PARAMS.Classes+1 : me, :) / sqrt(PARAMS.Constraint{2});
            A = A ./ repmat(normD, nr, 1);
            
            lambda = svds(W,1);
            mu = svds(A,1);
            
            W = (h*x' + lambda*W)/(x*x' + lambda*eye(nr));
            A = (q*x' + mu*A)/(x*x' + mu*eye(nr));
    end
    
    estimate(i) = I(1);
end
