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

function [D1, D2, X] = pair_dl_iter(Y, D1, D2, X)
%% Pairwise dictionary learning iteration
% INPUTS:
%   Y -- dense patch set stored as 3D matrix
%   D1 -- left dictionary
%   D2 -- right dictionary
%   X -- sparse representations
%   s -- sparsity constraint
%
% OUTPUTS:
%   D1 -- updated left dictionary
%   D2 -- updated right dictionary
%   X -- updated sparse representations
    n1 = size(D1,2);    % left atoms
    n2 = size(D2,1);    % right atoms
    N = size(Y,3);      % total number of patches
    R = zeros(size(Y));
    for k = 1:N
        R(:,:,k) = Y(:,:,k) - D1*X(:,:,k)*D2;    % residuals
    end

    for i = 1:n1
        % patches using the current atom
        [D2ind, I] = find(permute(X(i,:,:), [2 3 1]));
        if isempty(I)
            D1(:,i) = randn(size(D1,1), 1);
            D1(:,i) = D1(:,i) / norm(D1(:,i));
            %disp(['D1: replaced atom ' num2str(i)]);
            continue;
        end
        I = unique(I);
        D2ind = unique(D2ind);
        [R(:,:,I),D1(:,i), X(i,D2ind,I)] = pair_dl_left(R(:,:,I), D1(:,i), ...
            D2(D2ind,:), X(i,D2ind,I));
    end

    for i = 1:n2
        % patches using the current atom
        [D1ind, I] = find(permute(X(:,i,:), [1 3 2]));
        if isempty(I)
            D2(i,:) = randn(size(D2,2), 1);
            D2(i,:) = D2(i,:) / norm(D2(i,:));
            %disp(['D2: replaced atom ' num2str(i)]);
            continue;
        end
        I = unique(I);
        D1ind = unique(D1ind);
        [R(:,:,I),D2(i,:), X(D1ind,i,I)] = pair_dl_right(R(:,:,I), D2(i,:), ...
            D1(:,D1ind), X(D1ind,i,I));
    end
end

function [R, d1, X] = pair_dl_left(R,d1,D2,X)
    N = size(X,3);  % total number of patches using d1
    n2 = size(D2,1);% total number of atoms coupled with d1

    % residuals w/o current atoms
    for a = 1:n2
        for k = 1:N
            R(:,:,k) = R(:,:,k) + d1*X(1,a,k)*D2(a,:);
        end
    end

    % Update one variable at a time
    F = zeros(length(d1),1);
    for a = 1:n2
        % F = sum_a (sum_i R_i*x_i) * d2_a'
        F = F + sum(bsxfun(@times,R,X(1,a,:)),3)*D2(a,:)';
    end
    
    % For d1 maximize Tr(d1*F')
    d1 = F/norm(F);

    % For each line x_i we use the gradient of:
    % |R_i|_F^2 + - 2*D2*R_i'*d1*x_i + x_i*D2*D2'*x_i'
    for k = 1:N
        [~,I] = find(X(1,:,k));
        D2k = D2(I,:);
        X(1,I,k) = (D2k*D2k')\D2k*R(:,:,k)'*d1;
    end

    % Update the residuals
    for a = 1:n2
        for k = 1:N
            R(:,:,k) = R(:,:,k) - d1*X(1,a,k)*D2(a,:);
        end
    end
end

function [R, d2, X] = pair_dl_right(R,d2,D1,X)
    N = size(X,3);  % total number of patches using d2
    n1 = size(D1,2);% total number of atoms coupled with d2

    % residuals w/o current atoms
    for a = 1:n1
        for k = 1:N
            R(:,:,k) = R(:,:,k) + D1(:,a)*X(a,1,k)*d2;
        end
    end

    % Update one variable at a time
    F = zeros(1,length(d2));
    for a = 1:n1
        % F = sum_a d1_a' * sum_i R_i*x_i
        F = F + D1(:,a)'*sum(bsxfun(@times,R,X(a,1,:)),3);
    end

    % For d1 maximize Tr(d2*F')
    d2 = F/norm(F);

    % For each line x_i we use the gradient of:
    % |R_i|_F^2 + - 2*d2*R_i'*D1*x_i + x_i'*D1'*D1*x_i
    for k = 1:N
        [I,~] = find(X(:,1,k));
        D1k = D1(:,I);
        X(I,1,k) = (D1k'*D1k)\(d2*R(:,:,k)'*D1k)';
    end

    % Update the residuals
    for a = 1:n1
        for k = 1:N
            R(:,:,k) = R(:,:,k) - D1(:,a)*X(a,1,k)*d2;
        end
    end
end