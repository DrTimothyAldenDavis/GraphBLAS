function C = uminus (G)
%UMINUS negate a matrix.
% C = -G negates the entries of the matrix G.
%
% See also GrB.apply, GrB/minus, GrB/uplus.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

ghb = 0 ;     % 0 for GrB, 1 for GhB

C = GrB (gbmex_apply (ghb, '-', G)) ;

