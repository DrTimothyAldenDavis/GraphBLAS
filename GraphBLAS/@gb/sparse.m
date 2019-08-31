function C = sparse (G)
%SPARSE make a copy of a GraphBLAS sparse matrix.
% Since G is already sparse, C = sparse (G) simply makes a copy of G.
% Explicit zeros are not removed.  To remove them use
% C = gb.prune (G).
%
% See also gb/issparse, gb/full, gb.type, gb.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

C = gb (G) ;

