function s = gb_isvector (G)
%GB_ISVECTOR determine if the GraphBLAS matrix is a row or column vector,
% where G is the opaque struct of the GraphBLAS matrix.
% gb_isvector (G) is true for an m-by-n GraphBLAS matrix if m or n is 1.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

ghb = 1 ;     % 0 for GrB, 1 for GhB

[m, n] = gbmex_size (G) ;
s = (m == 1) || (n == 1) ;

