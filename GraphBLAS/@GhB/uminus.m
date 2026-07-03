function C = uminus (G)
%UMINUS negate a matrix.
% C = -G negates the entries of the matrix G.
%
% See also GhB.apply, GhB/minus, GhB/uplus.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

ghb = 1 ;     % 0 for GrB, 1 for GhB

C = gzb_apply (ghb, '-', G) ;

