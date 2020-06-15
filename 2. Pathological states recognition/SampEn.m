function SampEn = SampEn(x,m,r,tau)
% -------------------------------------------------------------------------
% Sample entropy (SampEn) of a univariate time series x
%
% Inputs:
%   x: univariate time series
%	m: embedding dimension
%   r: threshold (usually set to 0.15 * standard deviation of a time series)
%	tau: time delay factor (usually equal to 1)
% 
% Outputs:
%	SampEn: Distribution entropy value
%
% Reference:
%   [1] J. S. Richman and J. R. Moorman, "Physiological time-series analysis using approximate entropy and sample entropy", American Journal of Physiology-Heart and Circulatory Physiology, vol. 278, no. 6, pp.H2039-H2049, 2000.
%   [2] R. B. Govindan, J. D. Wilson, H. Eswaran, C. L. Lowery, and H. Prei��l, ��Revisiting sample entropy analysis,��, Physica A: Statistical Mechanics and its Applications, vol. 376, pp. 158?164, Mar. 2007.
%   [3] H. Azami and J. Escudero, "Refined Multiscale Fuzzy Entropy based on Standard Deviation for Biomedical Signal Analysis", Medical & Biological Engineering & Computing, 2016.
% -------------------------------------------------------------------------
 
% Check Inputs
narginchk(4, 4);

% -------------------------------------------------------------------------
% Version 1 (Ref.[3])
if nargin < 4, tau = 1; end    % parameter�� 4������ ���� ��
if tau > 1, x = downsample(x, tau); end
N = length(x);
P = zeros(1,2);
xMat = zeros(m+1,N-m);
for i = 1:m+1
    xMat(i,:) = x(i:N-m+i-1);
end

for k = m:m+1
    count = zeros(1,N-m);
    tempMat = xMat(1:k,:);
    
    for i = 1:N-k
        % calculate Chebyshev distance without counting self-matches
        dist = max(abs(tempMat(:,i+1:N-m) - repmat(tempMat(:,i),1,N-m-i)));
        
        % calculate the number of distance values that are less than the threshold r
        D = (dist <  r);
        count(i) = sum(D)/(N-m);
    end
    
    P(k-m+1) = sum(count)/(N-m);
end

SampEn = log(P(1)/P(2));

% -------------------------------------------------------------------------
% Version 2
% for k = m:m+1
%     N   = length(x) - (k-1)*tau;
%     rnt = embd(k,tau,x);
%     dv  = pdist(rnt, 'chebychev');
%     P(k-m+1) = length(find(dv < r))/length(dv);
% end
% 
% SampEn = log(P(1)/P(2));

end