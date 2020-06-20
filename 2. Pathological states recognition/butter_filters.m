function y = butter_filters(x,Fs,n,Wn,ftype)
% -------------------------------------------------------------------------
% Inputs:
%   x: siganl (array: returns filtered data for each column)
%   Fs: sampling rate
%   n: order
%   Wn: cut off frequency
%   ftype: 'high', 'stop', 'low', 'bandpass'
%
% Output:
%   y: filtered signal
% -------------------------------------------------------------------------

Fn = Fs/2;
[b, a] = butter(n, Wn/Fn, ftype);
y = filter(b,a,x);