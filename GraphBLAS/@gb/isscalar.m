function s = isscalar (G)
%ISSCALAR determine if the GraphBLAS matrix is a scalar.
% isscalar (G) is true for an m-by-n GraphBLAS matrix if m and n are 1.
%
% See also issparse, ismatrix, isvector, sparse, full, isa, gb.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

[m, n] = gbsize (G.opaque) ;
s = (m == 1) && (n == 1) ;

