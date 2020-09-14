%%  Parameters & data load
clear

m=2; t=1;
fs = 128; w = fs*5;

R=fs*1;
window=hamming(R);
N=2^8;
L=ceil(R/2);
overlap = R-L;

total_trial = 40;

class = {'HV','LV','HA','LA'};
[HA,LA,HV,LV] = Division_DEAP();

%%
num_sub = 32;

for loop=1:num_sub
    for class_num=1:length(class)
        if loop<10, sub = ['s0' num2str(loop)];
        else, sub = ['s' num2str(loop)];
        end
        fprintf(class{class_num})
        fprintf(' for Subject %d.. \n', loop);
        
        % --------------------------------------------------------------- %
        eval(['data = ' class{class_num} '.' sub ';']);        
        EEG = data(:,:,fs*3+1:end);                       
        psd_avg = zeros(size(EEG,1), 60/(w/fs), 32);
        
        for trial=1:size(EEG,1)
            sig(1:32,1:fs*60) = EEG(trial,:,:);           
            
            for ch=1:32
%                 eeg_base_win = sig_base(ch,:);
%                 [pxx_base, f_base] = pwelch(eeg_base_win, hamming(fs*1), [], N, fs);
%                 pxx_base = 10*log10(pxx_base);
                
%                 sig_nor = zscore(sig(ch,:));
%                 sd = sum(std(sig_nor,[],2));
%                 r = 0.15*sd;
                sig_nor = sig(ch,:);
                
                for seg=1:60*fs/w
                    eeg_win = sig_nor(w*(seg-1)+1:w*seg);
                    % -----------------------------------------------------%
                    % PSD feature
%                     [pxx_win,f] = pwelch(eeg_win, window, R/2, N, fs);
% %                     pxx_win = 10*log10(pxx_win);
%                     
% %                     pxx = pxx_win-pxx_base;
% %                     pxx = pxx_win;
% %                     psd_avg(trial,seg,ch) = mean(pxx(and(f>=band(1), f<=band(2))));
%                     psd_avg(trial,seg,ch) = bandpower(pxx_win,f,alpha_band,'psd');
                    % -----------------------------------------------------%
                    % Entropy feature
%                     ent = FuzEn(eeg_win,2,r,2,1);       % Fuzzy entropy
                    ent = pec(eeg_win,4,1);               % Permutation entropy
%                     ent = DistEn(eeg_win, 2, 1, 512);   % Distribution entropy
                    psd_avg(trial,seg,ch) = ent;
                end
            end           
        end
        
        % --------------------------------------------------------------- %
        % normalization
%         for i = 1:size(psd_avg,1)
%             tmp = psd_avg(i,:,:);
%             max_val = max(tmp(:));  min_val = min(tmp(:));
%             psd_avg(i,:,:) = (max_val-tmp)./(max_val-min_val);
%         end                        
        % --------------------------------------------------------------- %
        
%         eval([sub, '_Map_', class{class_num} ' = EEG_frame;']);
        eval([sub, '_Vec_', class{class_num} ' = psd_avg;']);
    end
end

%% save matrix
for s=1:num_sub
    if s<10, sub = ['s0' num2str(s)];
    else, sub = ['s' num2str(s)];
    end
    
    eval(['d1 = ' sub '_Vec_HV;']);
    eval(['d2 = ' sub '_Vec_LV;']);
    
    data = cat(1, d1, d2);
    
    % --------------------------------------------------------------- %
    % normalization
    data = (data-min(data(:))) / (max(data(:))-min(data(:)));
%     data = zscore(data);
    % --------------------------------------------------------------- %
    
    label = zeros(size(data,1),2);
    label(1:size(d1,1),1) = 1;
    label(1+size(d1,1):end,2) = 1;
    
    save(['../../../../python/Emotion_recognition/DEAP_data_using/dataset/Entropy/', sub, '_data.mat'], 'data')
    save(['../../../../python/Emotion_recognition/DEAP_data_using/dataset/Entropy/', sub, '_label.mat'], 'label')
    
    eval([sub '_tot = data;']);
end
