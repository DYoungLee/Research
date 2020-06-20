%% Example

%% Extract HRV time series (R-R interval) & Preprocessing

% ECG data load
load('ECG_sample.mat');
Fs = 250;

% Bandpass filtering
ECG_chf = butter_filters(ECG_chf, 250, 4, [0.1, 100], 'bandpass');
ECG_young = butter_filters(ECG_young, 250, 4, [0.1, 100], 'bandpass');
ECG_elderly = butter_filters(ECG_elderly, 250, 4,[0.1, 100], 'bandpass');

% R-peak detection using pan-tompkin algorithm
[~, r_peak, ~] = pan_tompkin(ECG_chf, Fs, 0);
for i = 1:length(r_peak)-1
    RR_chf(i) = r_peak(i+1) - r_peak(i);
end

[~, r_peak, ~] = pan_tompkin(ECG_young, Fs, 0);
for i = 1:length(r_peak)-1
    RR_young(i) = r_peak(i+1) - r_peak(i);
end

[~, r_peak, ~] = pan_tompkin(ECG_elderly, Fs, 0);
for i = 1:length(r_peak)-1
    RR_elderly(i) = r_peak(i+1) - r_peak(i);
end

% Preprocessing for R-R interval time series
RR_chf = RR_chf(or(RR_chf<Fs*2, RR_chf>Fs*0.2))/Fs;
RR_young = RR_young(or(RR_young<Fs*2, RR_young>Fs*0.2))/Fs;
RR_elderly = RR_elderly(or(RR_elderly<Fs*2, RR_elderly>Fs*0.2))/Fs;

%% Calculation of Multiscale Entropy methods

% Parameter setting
% Coarse-graining based Multiscale Entropy (MSE)
m_se = 2;
tau = 1;
r = 0.15;
scale = 10;

% Moving-averaging based Multiscale Distribution (MDE)
m_de = 2;
B = 512;

% Calculation of Multiscale Entropy methods
% CG: coarse-graining / MA: moving-averaging
w = 100;
for i=1:50
    MSE_chf(i,:) = MSE_CG(RR_chf(1+i*w:(i+1)*w), m_se, r, tau, scale, 1);
    MS_young(i,:) = MSE_CG(RR_young(1+i*w:(i+1)*w), m_se, r, tau, scale, 1);
    MSE_elderly(i,:) = MSE_CG(RR_elderly(1+i*w:(i+1)*w), m_se, r, tau, scale, 1);

    MDE_chf(i,:) = MDE_MA(RR_chf(1+i*w:(i+1)*w), m_de, B, scale);
    MDE_young(i,:) = MDE_MA(RR_young(1+i*w:(i+1)*w), m_de, B, scale);
    MDE_elderly(i,:) = MDE_MA(RR_elderly(1+i*w:(i+1)*w), m_de, B, scale);
end

% MSE results plot
figure('position', [0, 0, 1150, 450]);
x=1:scale; xlim([0 scale+1]);
xlabel('Scale', 'fontsize', 20); ylabel('Value', 'fontsize', 20);
set(gca,'FontSize',20)
hold on;
errorbar(mean(MSE_chf), std(MSE_chf,[],2), '-kd', 'linewidth', 2, 'markersize', 10);
errorbar(mean(MSE_elderly), std(MSE_elderly,[],2), '-bs', 'linewidth', 2, 'markersize', 10);
errorbar(mean(MSE_CG_young), std(MSE_CG_young,[],2), '-ro', 'linewidth', 2, 'markersize', 10);

% MDE results plot
figure('position', [0, 0, 1150, 450]);
x=1:scale; xlim([0 scale+1]);
xlabel('Scale', 'fontsize', 20); ylabel('Value', 'fontsize', 20);
set(gca,'FontSize',20)
hold on;
errorbar(mean(MDE_chf), std(MDE_chf,[],2), '-kd', 'linewidth', 2, 'markersize', 10);
errorbar(mean(MDE_elderly), std(MDE_elderly,[],2), '-bs', 'linewidth', 2, 'markersize', 10);
errorbar(mean(MDE_young), std(MDE_young,[],2), '-ro', 'linewidth', 2, 'markersize', 10);

%% Performance evalutaion
% Statistical analysis (p-value), classification accuracy, AUC value

[p_mse(1,:), acc_mse(1,:), AUC_mse(1,:)] = performance_eval(MSE_chf, MSE_elderly, scale);
[p_mse(2,:), acc_mse(2,:), AUC_mse(2,:)] = performance_eval(MSE_chf, MSE_young, scale);
[p_mse(3,:), acc_mse(3,:), AUC_mse(3,:)] = performance_eval(MSE_young, MSE_elderly, scale);

[p_mde(1,:), acc_mde(1,:), AUC_mde(1,:)] = performance_eval(MDE_chf, MDE_elderly, scale);
[p_mde(2,:), acc_mde(2,:), AUC_mde(2,:)] = performance_eval(MDE_chf, MDE_young, scale);
[p_mde(3,:), acc_mde(3,:), AUC_mde(3,:)] = performance_eval(MDE_young, MDE_elderly, scale);







