function e = nvals (G)
%NVALS the number of entries in a matrix.
% e = nvals (G) is the number of entries in a GraphBLAS matrix G.  A
% GraphBLAS matrix G may have explicit zero entries, and these are
% included in the count e.  Thus, nnz (G) <= GrB.nvals (G).
%
% See also GrB.entries, GrB.prune, GrB/nonzeros, GrB/size, GrB/numel,
% GrB/nnz.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

e = gbnvals (G) ;

