function e = gb_nnz (G)
%GB_NNZ the number of nonzeros in a GraphBLAS matrix.
% Implements e = nnz (G)

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

ghb = 0 ;     % 0 for GrB, 1 for GhB

% count entries in G and then subtract the number explicit zero entries
e = gbmex_nvals (G) - gbmex_nvals (GrB (gbmex_select (ghb, G, '==0'))) ;

