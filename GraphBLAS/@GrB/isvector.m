function s = isvector (G)
%ISVECTOR determine if the GraphBLAS matrix is a row or column vector.
% isvector (G) is true for an m-by-n GraphBLAS matrix if m or n is 1.
%
% See also GrB/issparse, GrB/ismatrix, GrB/isscalar, GrB/issparse,
% GrB/isfull, GrB/isa, GrB, GrB/size.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights
% Reserved. http://suitesparse.com.  See GraphBLAS/Doc/License.txt.

[m, n] = size (G) ;
s = (m == 1) || (n == 1) ;

