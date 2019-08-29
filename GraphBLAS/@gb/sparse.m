function S = sparse (G)
%SPARSE make a copy of a GraphBLAS sparse matrix.
% Since G is already sparse, S = sparse (G) simply makes a copy of G.
% Explicit zeros are not removed.
%
% See also issparse, full, gb.type, gb.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

S = gb (G) ;

