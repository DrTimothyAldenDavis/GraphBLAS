function e = gb_nnz (G)
%GB_NNZ the number of nonzeros in a GraphBLAS matrix.
% Implements e = nnz (G)

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

% count entries in G and then subtract the number explicit zero entries

% fprintf ('start with gb_nnz------------------------\n') ;
% fprintf ('get gbnvals (G):\n') ;
e1 = gbnvals (G) ;

% fprintf ('do select (G == 0):\n') ;
S = gbselect (G, '==0') ;

% fprintf ('get gbnvals (S):\n') ;
e2 = gbnvals (S) ;

e = e1 - e2 ;

% fprintf ('done with gb_nnz------------------------\n') ;

% e = gbnvals (G) - gbnvals (gbselect (G, '==0')) ;
