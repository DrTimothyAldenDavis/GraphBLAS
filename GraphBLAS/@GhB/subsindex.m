function I = subsindex (G)
%SUBSINDEX subscript index from a GraphBLAS matrix.
% I = subsindex (G) is an overloaded method used when the GraphBLAS
% matrix G is used to index into a non-GraphBLAS matrix A, for A(G).
%
% See also GrB/subsref, GrB/subsasgn.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

ghb = 1 ;     % 0 for GrB, 1 for GhB

I = gb_subsindex (ghb, G) ;

