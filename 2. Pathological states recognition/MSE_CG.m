function MSE = MSE_CG(x,m,r,tau,scale,nor)
% -------------------------------------------------------------------------
% Multiscale entropy (SampEn) of a univariate time series x
%
% Inputs:
%   x: univariate time series
%	m: embedding dimension
%   r: threshold (usually set to 0.15 * standard deviation of a time series)
%	tau: time delay factor (usually equal to 1)
%   nor: 1 (normalization O) or 0 (normalization X)
% 
% Outputs:
%	MSE: Multiscale entropy values for each scale
%
% Reference:
%   [1] M. Costa, A. L. Goldberger, and C.-K. Peng, “Multiscale Entropy Analysis of Complex Physiologic Time Series,” Physical Review Letters, vol. 89, no. 6, Jul. 2002.
%   [2] M. Costa, A. L. Goldberger, and C.-K. Peng, “Multiscale entropy analysis of biological signals,” Physical Review E, vol. 71, no. 2, Feb. 2005..
% -------------------------------------------------------------------------

% time series x is centered and normalised to standard deviation 1
if nor == 1
    x = x-mean(x);
    x = x./std(x);
end

MSE = zeros(1,scale);
for i=1:scale
    ms = Multi_mu(x,i);
    MSE(i)=SampEn(ms,m,r*std(x),tau);
end


function M_Data = Multi_mu(x,scale)
% -------------------------------------------------------------------------
% Coarse-graining procedure
%
% Inputs:
%       x: original univariate time series
%   scale: scale factor for multiscale process
% 
% Outputs:
%	M_Data: multiscale time series
% -------------------------------------------------------------------------

L = length(x);
J = fix(L/scale);

for i=1:J
    M_Data(i) = mean(x((i-1)*scale+1:i*scale));
end