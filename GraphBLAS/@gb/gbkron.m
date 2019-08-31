function Cout = gbkron (varargin)
%GB.GBKRON sparse Kronecker product.
%
% Usage:
%
%   Cout = gb.gbkron (op, A, B, desc)
%   Cout = gb.gbkron (Cin, accum, op, A, B, desc)
%   Cout = gb.gbkron (Cin, M, op, A, B, desc)
%   Cout = gb.gbkron (Cin, M, accum, op, A, B, desc)
%
% gb.gbkron computes the Kronecker product, T=kron(A,B), using the given
% binary operator op, in place of the conventional '*' operator for the
% MATLAB built-in kron.  See also C = kron (A,B), which uses the default
% semiring operators if A and/or B are gb matrices.
%
% All input matrices may be either GraphBLAS and/or MATLAB matrices, in
% any combination.  Cout is returned as a GraphBLAS matrix, by default;
% see 'help gb/descriptorinfo' for more options.
%
% T is then accumulated into C via C<#M,replace> = accum (C,T).
%
% See also kron, gb/kron.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

[args is_gb] = get_args (varargin {:}) ;
if (is_gb)
    Cout = gb (gbkronmex (args {:})) ;
else
    Cout = gbkronmex (args {:}) ;
end

