function C = eps (G)
%EPS Spacing of floating-point numbers in a GraphBLAS matrix.

% FUTURE: this could be much faster as a mexFunction.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

if (~isfloat (G))
    error ('Type must be ''single'', ''double'', or ''complex''') ;
end
[m, n] = size (G) ;
[i, j, x] = gb.extracttuples (full (G), struct ('kind', 'zero-based')) ;
C = gb.build (i, j, eps (x), m, n) ;

