function C = int64 (G)
%INT64 cast a GraphBLAS matrix to MATLAB full int64 matrix.
% C = int64 (G) typecasts the gb matrix G to a MATLAB full int64 matrix.
% The result C is full since MATLAB does not support sparse int64
% matrices.
%
% To typecast the matrix G to a GraphBLAS sparse int64 matrix instead,
% use C = gb (G, 'int64').
%
% See also gb, double, complex, single, logical, int8, int16, int32,
% uint8, uint16, uint32, and uint64.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

C = gbfull (G.opaque, 'int64') ;

