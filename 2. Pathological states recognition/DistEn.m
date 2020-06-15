function DistEn = DistEn(x,m,tau,B)
% -------------------------------------------------------------------------
% Distribution Entropy (DistEn) of a univariate time series x
%
% Inputs:
%   x: univariate time series
%	m: embedding dimension
%	tau: time delay factor
%	B: number of histogram bins
% 
% Outputs:
%	DistEn: Distribution entropy value
%
% Reference:
%   [1] P. Li, C. Liu, K. Li, D. Zheng, C. Liu, and Y. Hou, "Assessing the complexity of short-term heartbeat interval series by distribution entropy", Med. Biol. Eng. Comput., vol. 53, no. 1, pp. 77?87, Jan. 2015.
% -------------------------------------------------------------------------
 
% Check Inputs
narginchk(4, 4);

% Rescaling
x = (x - min(x)) ./ (max(x) - min(x));
 
% Reconstruction
% Version 1
N   = length(x) - (m-1)*tau;
ind = hankel(1:N, N:length(x));
rnt = x(ind(:, 1:tau:end));
% Version 2
% rnt = embd(m,tau,x);

% Distance matrix
dv  = pdist(rnt, 'chebychev');      % chebychev distance
                                    % (N-1)N/2 (1+2+...+N-1)
 
% Esimating probability density by histogram
num  = hist(dv, linspace(0, 1, B));
freq = num./sum(num);
 
% Normalized DistEn calculation
DistEn = -sum(freq.*log2(freq+eps)) ./ log2(B);
