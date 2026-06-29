function e = nnz (G)
%NNZ the number of nonzeros in a matrix.
% e = nnz (G) is the number of nonzeros in a GraphBLAS matrix G.  A
% GraphBLAS matrix G may have explicit zero entries, but these are
% excluded from the count e.  Thus, nnz (G) <= GrB.entries (G).
%
% See also GrB.entries, GrB.prune, GrB/nonzeros, GrB/size, GrB/numel.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

ghb = 1 ;     % 0 for GrB, 1 for GhB

e = gb_nnz (ghb, G) ;

