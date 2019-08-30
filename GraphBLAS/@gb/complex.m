function C = complex (A, B)
%COMPLEX cast a GraphBLAS matrix to MATLAB sparse double complex matrix.
% C = complex (G) will typecast the GraphBLAS matrix G to into a MATLAB
% sparse logical matrix.
%
% To typecast the matrix G to a GraphBLAS sparse complex matrix instead,
% use C = gb (G, 'complex').
%
% See also cast, gb, double, single, logical, int8, int16, int32, int64,
% uint8, uint16, uint32, and uint64.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

error ('complex type not yet supported') ;

