function Multi_vec = embd(M,tau,mx)
% -------------------------------------------------------------------------
% Creates multivariate delay embedded vectors with embedding
% Inputs:
%   M: row vector [m1 m2 ...m_nvar]
%   tau: row vector [tau1 tau2....tau_nvar]
%   (nvar is the number of channel)
%   mx: multivariate time series (a matrix of size nvar*nsamp)
%
% Outputs:
%   Multi_vec: multivariate delay embedded vectors
%
% Reference:
%   [1] M. U. Ahmed and D. P. Mandic, "Multivariate multiscale entropy analysis", IEEE Signal Processing Letters, vol. 19, no. 2, pp.91-94, 2012
%   [2] http://www.commsp.ee.ic.ac.uk/∼mandic/research/Complexity Stuff.htm.
% -------------------------------------------------------------------------

[nvar,nsamp]=size(mx);
Multi_vec=[];

for j=1:nvar
    for i=1:nsamp-(max(M)*max(tau))
        temp(i,:)=mx(j,i:tau(j):i+M(j)*tau(j)-1);
    end
    Multi_vec=horzcat(Multi_vec,temp);
    temp=[];
end






