function [p, accuracy, AUC] = performance_eval(data1, data2, scale)
% -------------------------------------------------------------------------
% Binary classification for multiscale entropy features
%
% Inputs:
%   data1, data2: multiscale entropy values (samples x scale)
%	scale: scale factor
% 
% Outputs:
%	p: statistical analysis results, p-value
%   accuracy: classisification accuracy
%   AUC: area under ROC curve
% -------------------------------------------------------------------------
%%
data1(data1==inf) = NaN;
data2(data2==inf) = NaN;

X = vertcat(data1, data2);
Y = zeros(size(X,1), 1);
Y(1:size(data1,1)) = 1;

%%
for s = 1:scale
    % p-value
    p(s,1) = ranksum(data1(s), data2(s));
    % p(scale,1) = ranksum(tmp1(randsample(size(tmp1,1),100),scale), tmp2(randsample(size(tmp2,1),100),scale));
    
    % 5-fold Cross Validation
    accuracy(s,1) = mean(k_fold_SVM(X(:,1:s),Y,5));
    
    % ROC curve using SVM classifier
    mdlSVM = fitcsvm(X(:,1:s), boolean(Y), 'Standardize', true, 'BoxConstraint', 10,'KernelFunction','RBF');
    mdlSVM = fitPosterior(mdlSVM);
    [~,score_svm] = resubPredict(mdlSVM);
    [Xsvm, Ysvm, ~, AUC(s,1)] = perfcurve(Y,score_svm(:,mdlSVM.ClassNames),'true');
end
