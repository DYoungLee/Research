%% Setting
% DEAP data load
[HA,LA,HV,LV] = Division_DEAP();
class = {'HA','LA', 'HV', 'LV'};

% Parameters for entropy methods
B=512; c=6; r=0.2;
fs=128; w=fix(fs*3);                % 3s overlapping window
ch = [8,26]; num_ch=length(ch);     % T7:8, T8:26
clear M t
for i=1:length(ch)
    M(i) = 2; t(i) = 1;
end
n_imf = 10;

%% Methods
clear eeg_sig sig
num_sub = 32;

for class_num=1:length(class)
    for loop=1:num_sub
        if loop<10, sub = ['s0' num2str(loop)];
        else, sub = ['s' num2str(loop)];
        end
        fprintf(class{class_num});
        fprintf(' for Subject %d.. \n', loop);
        eval(['data = ' class{class_num} '.' sub ';']);
        eval(['sub' num2str(loop) '.' class{class_num} '= [];']);
        
        if isempty(data), continue
        end
        
        clear cimf memd_sig        
        for N=1:size(data,1)                     
        %%%% MEMD calculate %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % last 30 seconds
            eeg_sig(1:num_ch,:) = data(N,ch,fs*33+1:end);
            memd_sig = memd(eeg_sig);
            eval(['memd_DEAP_' class{class_num} '.' sub '.trial' num2str(N) ' = memd_sig;']);
       
        %%%% Various scale methods %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
            % Cumulitive IMF 
            for i=1:size(memd_sig,2)
                cimf(1:num_ch,i,:) = sum(memd_sig(:,1:i,:),2);                
            end
        % -----------------------------------------------------------------
            % Fine-to-coarse (likes high-pass filter)
%             for i=1:size(memd_sig,2)
%                 cimf(1:num_ch,i,:) = sum(memd_sig(:,i:end,:),2);                
%             end
        % -----------------------------------------------------------------    
            % Coarse-to-fine (likes low-pass filter)
%             for i=size(memd_sig,2):-1:1
%                 cimf(1:num_ch,size(memd_sig,2)-i+1,:) = sum(memd_sig(:,1:i,:),2);                
%             end
        % -----------------------------------------------------------------     
            
        %%%% Multivariate entropy methods %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            entropy_val=[];
            for i=1:n_imf
                len = size(cimf,3);
                clear sig
                try 
                    sig(1:num_ch,:) = cimf(:,i,:);
                catch
                    break
                end
            % -------------------------------------------------------------
            % Normalization 
%                 sig = (sig - min(sig,[],2)) ./ (max(sig,[],2) - min(sig,[],2));
                sig = zscore(sig')';
                sd = sum(std(sig,[],2));
                r = 0.15*sd;      
            % -------------------------------------------------------------      
            % Various entropy calculation 
                L = size(sig,2);            
                
                j=1;
                sigma = std(sig,[],2);
                mu = mean(sig,2);
                while w/2*(j-1)+w<=L
                    entropy_val(j,i) = mvSE(sig(:,w/2*(j-1)+1:w/2*(j-1)+w),M,r,t);
%                     entropy_val(j,i) = mvFE(sig(:,w/2*(j-1)+1:w/2*(j-1)+w),M,r,2,t);                    
%                     entropy_val(j,i) = mvDispEn_ms(sig(:,w/2*(j-1)+1:w/2*(j-1)+w),M,c,t,mu,sigma);
%                     entropy_val(j,i) = JDistEn(sig(:,w/2*(j-1)+1:w/2*(j-1)+w),M,t); 
%                     entropy_val(j,i) = joint_pe(sig(:,w/2*(j-1)+1:w/2*(j-1)+w)',4,1);
%                     entropy_val(j,i) = mvFE(sig(:,w/2*(j-1)+1:w/2*(j-1)+w),M,r,2,t);
                    j=j+1;
                end          
                % ---------------------------------------------------------               
            end
            
            % Concatenate all entropy values
            eval(['tmp = sub' num2str(loop) '.' class{class_num} ';']);
            tmp(size(tmp,1)+1:size(tmp,1)+size(entropy_val,1),1:size(entropy_val,2)) = entropy_val;
            eval(['sub' num2str(loop) '.' class{class_num} ' = tmp;']);
        end        
    end
end
