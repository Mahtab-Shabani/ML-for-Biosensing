% unsupervised_x1_x2.m
% Unsupervised detection of EMG vs EEG using only x1 and x2
% Compatible with MATLAB 2014a

clc; clear; close all;
fprintf('Running unsupervised_x1_x2...\n');

%% 0. Settings
load('data.mat');   % expects variables x1, x2 (and others)
fs = 256;           % sampling freq - change if different
win_sec = 2;        % window length in seconds
step_sec = 1;       % step (overlap)
win = win_sec * fs;
step = step_sec * fs;

% choose segment for EMG candidate if desired (the original script used)
% EMG1 = x1(1001:6000);  % optional - here we will use entire x1 and x2
sig_names = {'x1','x2'};

%% 1. Build windows from x1 and x2
all_feats = [];
all_labels = []; % helper: remember which signal each window came from
windows = {};

for s = 1:length(sig_names)
    name = sig_names{s};
    if ~exist(name,'var')
        error('Variable %s not found in data.mat', name);
    end
    sig = eval(name);
    sig = sig(:); % ensure column

    N = length(sig);
    starts = 1:step:(N - win + 1);
    for k = 1:length(starts)
        idx = starts(k):(starts(k)+win-1);
        seg = sig(idx);

        % ---- features ----
        % time-domain
        MAV = mean(abs(seg));
        RMS = sqrt(mean(seg.^2));
        WL = sum(abs(diff(seg)));         % waveform length
        ZCR = sum(abs(diff(sign(seg)))) / length(seg);

        % spectral (use pwelch)
        nfft = 512;
        [Pxx,f] = pwelch(seg, hamming(256), 128, nfft, fs);
        % band powers
        bp_delta = bandpower_from_psd(Pxx,f,[1 4]);
        bp_theta = bandpower_from_psd(Pxx,f,[4 8]);
        bp_alpha = bandpower_from_psd(Pxx,f,[8 13]);
        bp_beta  = bandpower_from_psd(Pxx,f,[13 30]);
        bp_hi    = bandpower_from_psd(Pxx,f,[40 200]); % high freq energy => EMG

        % spectral entropy
        specEnt = spectral_entropy(Pxx);

        % kurtosis
        K = kurtosis(seg);

        feat = [MAV, RMS, WL, ZCR, bp_delta, bp_theta, bp_alpha, bp_beta, bp_hi, specEnt, K];
        all_feats = [all_feats; feat];
        all_labels = [all_labels; s]; % 1 for x1, 2 for x2
        windows{end+1} = [name, sprintf(': %d-%d', idx(1), idx(end))]; %#ok<SAGROW>
    end
end

feature_names = {'MAV','RMS','WL','ZCR','bp_d','bp_t','bp_a','bp_b','bp_hi','specEnt','kurt'};

%% 2. Normalize features
mu = mean(all_feats,1);
sigma = std(all_feats,0,1) + eps;
X = (all_feats - repmat(mu,size(all_feats,1),1)) ./ repmat(sigma,size(all_feats,1),1);

%% 3. Dimensionality reduction (PCA) for visualization
[coeff,score,~] = pca(X);

%% 4. k-means clustering (k=2)
k = 2;
rng(1); % for reproducibility
[idx, C] = kmeans(X, k, 'Replicates', 20);

%% 5. Determine which cluster is EMG
% compute mean high-frequency power (bp_hi) per cluster
bp_hi_col = find(strcmp(feature_names,'bp_hi'));
cluster_bp_hi = zeros(k,1);
for c = 1:k
    cluster_bp_hi(c) = mean(all_feats(idx == c, bp_hi_col));
end
[~, emg_cluster] = max(cluster_bp_hi); % cluster with highest bp_hi is EMG
eeg_cluster = setdiff(1:k, emg_cluster);

%% 6. Map results back to signals
is_emg_window = (idx == emg_cluster);
% count per original signal
for s = 1:length(sig_names)
    cnt_total = sum(all_labels == s);
    cnt_emg = sum(all_labels == s & is_emg_window);
    fprintf('%s: total windows=%d, windows labeled EMG=%d\n', sig_names{s}, cnt_total, cnt_emg);
end

% decide signal-level label: majority of windows
signal_label = zeros(length(sig_names),1);
for s=1:length(sig_names)
    cnt_emg = sum(all_labels==s & is_emg_window);
    if cnt_emg > sum(all_labels==s)/2
        signal_label(s) = 1; % EMG
    else
        signal_label(s) = 0; % EEG-like
    end
end

for s=1:length(sig_names)
    if signal_label(s)==1
        fprintf('=> %s identified as EMG (by majority of windows)\n', sig_names{s});
    else
        fprintf('=> %s identified as EEG-like\n', sig_names{s});
    end
end

%% 7. Save results
save('unsupervised_x1_x2_results.mat','all_feats','feature_names','X','idx','C','mu','sigma','windows','all_labels','is_emg_window','signal_label');
T = array2table(all_feats,'VariableNames',feature_names);
T.Signal = all_labels;
T.Cluster = idx;
writetable(T,'unsupervised_x1_x2_features.csv');

%% 8. Plots
figure;
gscatter(score(:,1), score(:,2), idx);
xlabel('PC1'); ylabel('PC2'); title('PCA of windows (x1,x2) colored by cluster');
saveas(gcf,'unsupervised_x1_x2_pca.png');

figure;
bar(cluster_bp_hi);
xlabel('Cluster'); ylabel('Mean high-band power (40-200Hz)');
title('Cluster high-frequency power (EMG indicator)');
saveas(gcf,'unsupervised_x1_x2_bp_hi.png');

fprintf('Done. Results saved: unsupervised_x1_x2_results.mat and CSV/PNG files.\n');
