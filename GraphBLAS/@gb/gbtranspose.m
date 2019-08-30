function Cout = gbtranspose (varargin)
%GB.GBTRANSPOSE transpose a sparse matrix
%
% Usage:
%
%   Cout = gb.gbtranspose (A, desc)
%   Cout = gb.gbtranspose (Cin, accum, A, desc)
%   Cout = gb.gbtranspose (Cin, M, A, desc)
%   Cout = gb.gbtranspose (Cin, M, accum, A, desc)
%
% The descriptor is optional.  If desc.in0 is 'transpose', then C<M>=A or
% C<M>=accum(C,A) is computed, since the default behavior is to transpose
% the input matrix.
%
% See also transpose, ctranspose.

[args is_gb] = get_args (varargin {:}) ;
if (is_gb)
    Cout = gb (gbtransposemex (args {:})) ;
else
    Cout = gbtransposemex (args {:}) ;
end

