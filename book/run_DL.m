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

%% Basic Algorithms: DL on images -- specific in-depth tests
clear; clc; close all; fclose all; format compact;
%%-------------------------------------------------------------------------
m = 64;                 % problem dimension
nn = 128:64:512;        % number of atoms in the dictionary
N = 4000;               % number of training signals
ss = [4 6 8 10 12];     % sparsity constraint
K = 500;                % DL iterations
rounds = 10;            % test rounds

generate_data = true;   % generate initial data
ots = '20170818164513'; % else use data timestamp, copy from fig_DL_3_init

% Dictionary update routines
updates = {'MOD', 'sgk', 'ksvd', 'aksvd', 'nsgk', 'paksvd', 'pnsgk'};
% Unused atoms replacement strategy
replatoms = 'random';

% Data output
datadir = 'data\';
dataprefix = 'fig_3_DL';

% Training images
imdir = 'img\';
images = {'barbara.png', 'boat.png', 'house.png', 'lena.png', 'peppers.png'};
%%-------------------------------------------------------------------------
addpath(genpath('DL'));

timestamp = datestr(now, 'yyyymmddHHMMss');
if generate_data
    Y = cell(rounds,1);
    D0 = cell(rounds,length(nn),1);
    for r = 1:rounds
        Y{r} = mkimgsigs(imdir,images,m,N,'distinct');
        for i = 1:length(nn)
            D0{r,i} = normc(randn(m,nn(i)));
        end
    end
    save([datadir 'fig_3_DL_init-' timestamp '.mat'], 'Y', 'D0');
else
    load([datadir 'fig_3_DL_init-' ots '.mat'], 'Y', 'D0');
end

methods = length(updates);

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
fprintf('\n\tParameters: m=%d N=%d K=%d rounds=%d replatoms=%s', ...
    m, N, K, rounds, replatoms);

for i = 1:length(nn)
    n = nn(i);
for s = ss
    fprintf('\n(n=%d,s=%d): ', n, s); 
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
                'replatoms', replatoms);
            time_end = clock;
            times(r,j) = etime(time_end,time_start);  
        end
    end
    %% Write out data
    matfile = sprintf('%s%s-m%d-n%d-N%d-s%d-K%d-%s.mat', ...
         datadir, dataprefix, m, n, N, s, K, timestamp);    
    save(matfile,'updates', 'replatoms', 'm', 'n', 'N', 's', 'K', ...
        'Dall', 'Xall', 'errs', 'criteria','times');
end % sparsity loop
end % atoms loop
