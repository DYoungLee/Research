function [accuracy, CM] = k_fold_SVM(X,Y,k)
% -------------------------------------------------------------------------
% Input value:
%   X: Feature vectors (Observation values X Features)
%   Y: Labels
%   k: k-fold
%
% Output value:
%   auc: Accuracy of each fold
%   CM: Confusion matrix
%--------------------------------------------------------------------------
cvFolds = crossvalind('Kfold', Y, k);           % Index of k-fold CV
cp = classperf(Y);                              % init performance tracker

for i = 1:k                                     % for each fold
    testIdx = (cvFolds == i);                   % Index for test
    trainIdx = ~testIdx;                        % Index for training

    svmModel = fitcsvm(X(trainIdx,:), Y(trainIdx), 'Standardize',true,...
        'BoxConstraint', 10, 'KernelFunction','RBF');
  
    % prediction of test dataset
    pred = predict(svmModel, X(testIdx,:));
    
    % accuracy calculation
    accuracy(i) = length(find(Y(testIdx)-pred == 0)) / length(pred);
    
    % evaluate and update performance object
    cp = classperf(cp, pred, testIdx);
end

% total accuracy
auc_mean=cp.CorrectRate;

% get confusion matrix
% column: actual, row: predicted, last-row: unclassified
CM = cp.CountingMatrix;