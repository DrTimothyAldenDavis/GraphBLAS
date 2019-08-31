function Cout = gbtranspose (varargin)
%GB.GBTRANSPOSE transpose a sparse matrix.
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
% All input matrices may be either GraphBLAS and/or MATLAB matrices, in
% any combination.  Cout is returned as a GraphBLAS matrix, by default;
% see 'help gb/descriptorinfo' for more options.
%
% See also transpose, ctranspose.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

[args is_gb] = get_args (varargin {:}) ;
if (is_gb)
    Cout = gb (gbtransposemex (args {:})) ;
else
    Cout = gbtransposemex (args {:}) ;
end

