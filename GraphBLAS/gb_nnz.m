function e = gb_nnz (ghb, G)
%GB_NNZ the number of nonzeros in a GraphBLAS matrix.  Not user-callable.
% Implements e = nnz (G)

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

% count entries in G and then subtract the number explicit zero entries
e = gbmex_nvals (G) - gbmex_nvals (gzb_select (ghb, G, '==0')) ;

