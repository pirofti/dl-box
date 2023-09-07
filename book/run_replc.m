% Copyright (c) 2016-2018 Paul Irofti <paul@irofti.net>
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

%% Basic Algorithms: DL atom replacement
clear; clc; close all; fclose all; format compact;
%%-------------------------------------------------------------------------
m = 64;                 % problem dimension
nn = [128 192 256 512]; % number of atoms in the dictionary
N = 2048;               % number of training signals
s = 8;                  % sparsity constraint
K = 50;                 % DL iterations
rounds = 10;            % test rounds
ots = '20180422183235'; % data timestamp, copy from fig_3_3_init file

% Dictionary update routines
updates = {'aksvd'};
% Unused atoms replacement strategy
repl = {'no', 'random', 'worst', 'zero'};
extra = {'', '', '', 'worstrep'};

% Data output
datadir = 'data\';
dataprefix = 'fig_3_4_replc';

% Training images
imdir = 'img\'; %racheta
images = {'barbara.png', 'boat.png', 'house.png', 'lena.png', 'peppers.png'};
%%-------------------------------------------------------------------------
addpath(genpath('DL'));

load([datadir 'fig_3_3_init-' ots '.mat'], 'Y', 'D0');

methods = length(updates);

% Center data
for r=1:rounds
    Ymean = mean(Y{r});
    Y{r} = Y{r} - repmat(Ymean, m, 1);    % 0-mean
end
   
% Header
fprintf('Basic Algorithms');
fprintf('\n\tImages: ');
for i = 1:length(images)
    fprintf('%s ', images{i});
end
fprintf('\n\tMethods: ');
for i = 1:methods
    fprintf('%s ', updates{i});
end
fprintf('\n\tParameters: m=%d N=%d K=%d rounds=%d', m, N, K, rounds);

for i = 1:length(nn)
    n = nn(i);
for k = 1:length(repl)
    replatoms = repl{k};
    postopts = extra{k};
    fprintf('\n(n=%d,repl=%s): ', n, replatoms);
    Dall = zeros(rounds,methods, m, n);
    Xall = zeros(rounds,methods, n, N);
    errs = zeros(rounds,methods, K);
    criteria = zeros(rounds,methods, K);
    times = zeros(rounds,methods);
    for r = 1:rounds
        fprintf('%d', mod(r, 10)); 
        %% Rounds
        Yr = zeros(m,N);
        D0r = zeros(m,n);       
        D0r(:,:) = D0{r,i};
        Yr(:,:) = Y{r};

        for j = 1:methods
            fprintf('%s', updates{j}(1));
            time_start = clock;
            [Dall(r,j,:,:), Xall(r,j,:,:), errs(r,j,:)] = ...
                DL(Yr, D0r, s, K, str2func(updates{j}), ...
                'replatoms', replatoms, 'postopts', postopts);
            time_end = clock;
            times(r,j) = etime(time_end,time_start);  
        end
    end
    %% Write out data
    matfile = sprintf('%s%s-m%d-n%d-N%d-s%d-K%d-%s-%s.mat', ...
         datadir, dataprefix, m, n, N, s, K, replatoms, ots);    
    save(matfile,'updates', 'replatoms', 'm', 'n', 'N', 's', 'K', ...
        'Dall', 'Xall', 'errs', 'criteria','times','Y');
end % repl loop
end % atoms loop
