%% DEAP data
function [HA,LA,HV,LV] = Division_DEAP()
    for loop=1:32
        if loop<10, sub = ['s0' num2str(loop)];
        else, sub = ['s' num2str(loop)];
        end

        fprintf('Subject %d.. \n', loop);
        load(['../../../Dataset/Emotion/DEAP/', sub, '.mat'])
        % data: 40 x 40 x 8064 (trial x channel x data)
        % labels: 40 x 4 (trial x label (valence, arousal, dominance, liking))
    
        eval(['HA.' sub '= data(find(labels(:,2)>=5),1:32,:);'])
        eval(['LA.' sub '= data(find(labels(:,2)<5),1:32,:);'])
        eval(['HV.' sub '= data(find(labels(:,1)>=5),1:32,:);'])
        eval(['LV.' sub '= data(find(labels(:,1)<5),1:32,:);'])
    end
end