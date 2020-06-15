function MDE = MDE_MA(x,m,B,scale)
% -------------------------------------------------------------------------
% Moving-averaging-based Multiscale Distribution Entropy (MDE) of a univariate time series x
%
% Inputs:
%       x: original univariate time series
%       m: embedding dimension
%       B: number of histogram bins
%   scale: scale factor for multiscale process
% 
% Outputs:
%	MDE: Multiscale Distribution entropy values for each scale
%
% Reference:
%   [1] S.-D. Wu, C.-W. Wu, K.-Y. Lee, and S.-G. Lin, “Modified multiscale entropy for short-term time series analysis,” Physica A: Statistical Mechanics and its Applications, vol. 392, no. 23, pp. 5865–5873, Dec. 2013.
%   [2] D.-Y. Lee and Y.-S. Choi, “Multiscale Distribution Entropy Analysis of Short-Term Heart Rate Variability,” Entropy, vol. 20, no. 12, p. 952, Dec. 2018.
%   [3] D.-Y. Lee and Y.-S. Choi, “Multiscale Distribution Entropy Analysis of Heart Rate Variability Using Differential Inter-Beat Intervals,” IEEE Access, vol. 8, pp. 48761–48773, 2020.
% -------------------------------------------------------------------------

MDE = zeros(1,scale);
for i = 1:scale
    ms = Multi_MA(x,i);
    MDE(i) = MDE(i) + DistEn(ms, m, i, B);
end

end


function M_Data = Multi_MA(x,scale)
% -------------------------------------------------------------------------
% Moving-averaging procedure
%
% Inputs:
%       x: original univariate time series
%   scale: scale factor for multiscale process
% 
% Outputs:
%	M_Data: multiscale time series
% -------------------------------------------------------------------------

N = length(x);
J = N-scale+1;

for i=1:J
    M_Data(i) = mean(x(i:i+scale-1));
end

end