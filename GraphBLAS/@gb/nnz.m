function e = nnz (G)
%NNZ the number of nonzeros in a GraphBLAS matrix.
% e = nnz (G) is the number of nonzeros in a GraphBLAS matrix G.  A GraphBLAS
% matrix G may have explicit zero entries, but these are excluded from the
% count e.  Thus, nnz (G) <= gb.entries (G).
%
% See also gb.entries, gb.prune, nonzeros, size, numel.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

e = gb.entries (G) - gb.entries (gb.select ('==0', G)) ;

