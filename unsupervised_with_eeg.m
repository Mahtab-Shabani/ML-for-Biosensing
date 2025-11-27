% unsupervised_with_eeg.m
% Unsupervised detection of EMG vs EEG using x1, x2 plus binned EEG bands
% Compatible with MATLAB 2014a

clc; clear; close all;
fprintf('Running unsupervised_with_eeg...\n');

%% 0. Settings and load
load('data.mat');   % expects: x1, x2, dalt, thalt, alpha, belta
fs = 256;
win_sec = 2;
step_sec = 1;
win = win_sec*fs;
step = step_sec*fs;

% signals to use (raws + band signals)
sig_list = {'x1','x2','dalt','thalt','alpha','belta'};
nSigs = length(sig_list);

all_feats = [];
all_meta = {}; % store (signalName, startIdx, endIdx)

for s = 1:nSigs
    name = sig_list{s};
    if ~exist(name,'var')
        warning('Variable %s not found — skipping', name);
        continue;
    end
    sig = eval(name);
    sig = sig(:);
    N = length(sig);
    starts = 1:step:(N - win + 1);
    for k = 1:length(starts)
        idx = starts(k):(starts(k)+win-1);
        seg = sig(idx);

        % features
        MAV = mean(abs(seg));
        RMS = sqrt(mean(seg.^2));
        WL = sum(abs(diff(seg)));
        ZCR = sum(abs(diff(sign(seg)))) / length(seg);

        nfft = 512;
        [Pxx,f] = pwelch(seg, hamming(256), 128, nfft, fs);
        bp_d = bandpower_from_psd(Pxx,f,[1 4]);
        bp_t = bandpower_from_psd(Pxx,f,[4 8]);
        bp_a = bandpower_from_psd(Pxx,f,[8 13]);
        bp_b = bandpower_from_psd(Pxx,f,[13 30]);
        bp_30_80 = bandpower_from_psd(Pxx,f,[30 80]);
        bp_hi = bandpower_from_psd(Pxx,f,[40 200]);

        specEnt = spectral_entropy(Pxx);
        K = kurtosis(seg);

        feat = [MAV,RMS,WL,ZCR,bp_d,bp_t,bp_a,bp_b,bp_30_80,bp_hi,specEnt,K];
        all_feats = [all_feats; feat];
        all_meta{end+1,1} = name;
        all_meta{end,2} = idx(1);
        all_meta{end,3} = idx(end);
    end
end

feature_names = {'MAV','RMS','WL','ZCR','bp_d','bp_t','bp_a','bp_b','bp_30_80','bp_hi','specEnt','kurt'};

%% Normalize
mu = mean(all_feats,1); sigma = std(all_feats,0,1) + eps;
X = (all_feats - repmat(mu,size(all_feats,1),1)) ./ repmat(sigma,size(all_feats,1),1);

%% PCA
[coeff,score,~] = pca(X);

%% kmeans (k=2)
k = 2;
rng(1);
[idx, C] = kmeans(X, k, 'Replicates', 20);

%% Identify EMG cluster by bp_hi
bp_hi_col = find(strcmp(feature_names,'bp_hi'));
cluster_bp_hi = zeros(k,1);
for c=1:k
    cluster_bp_hi(c) = mean(all_feats(idx==c, bp_hi_col));
end
[~, emg_cluster] = max(cluster_bp_hi);
eeg_cluster = setdiff(1:k, emg_cluster);

%% Summarize per original signal
unique_signals = unique(all_meta(:,1),'stable');
signal_summary = struct();
for i = 1:length(unique_signals)
    nm = unique_signals{i};
    mask = strcmp(all_meta(:,1), nm);
    total = sum(mask);
    emgwin = sum(mask & idx==emg_cluster);
    signal_summary.(nm).total_windows = total;
    signal_summary.(nm).emg_windows = emgwin;
    % decide majority
    signal_summary.(nm).isEMG = emgwin > (total/2);
end

% print
disp('Signal-level decisions:');
for i = 1:length(unique_signals)
    nm = unique_signals{i};
    if signal_summary.(nm).isEMG
        fprintf('%s -> EMG (EMG windows %d / %d)\n', nm, signal_summary.(nm).emg_windows, signal_summary.(nm).total_windows);
    else
        fprintf('%s -> EEG-like (EMG windows %d / %d)\n', nm, signal_summary.(nm).emg_windows, signal_summary.(nm).total_windows);
    end
end

%% Save results
save('unsupervised_with_eeg_results.mat','all_feats','feature_names','X','idx','C','mu','sigma','all_meta','signal_summary');
T = array2table(all_feats,'VariableNames',feature_names);
T.Signal = all_meta(:,1);
T.Start = cell2mat(all_meta(:,2));
T.End   = cell2mat(all_meta(:,3));
T.Cluster = idx;
writetable(T,'unsupervised_with_eeg_features.csv');

%% Plots
figure;
gscatter(score(:,1), score(:,2), idx);
xlabel('PC1'); ylabel('PC2'); title('PCA (all signals) colored by cluster');
saveas(gcf,'unsupervised_with_eeg_pca.png');

figure;
bar(cluster_bp_hi);
xlabel('Cluster'); ylabel('Mean high-band power (40-200Hz)');
title('Cluster high-frequency power (EMG indicator)');
saveas(gcf,'unsupervised_with_eeg_bp_hi.png');

fprintf('Done. Results saved: unsupervised_with_eeg_results.mat and CSV/PNG files.\n');

