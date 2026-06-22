function C = transpose (G)
%TRANSPOSE C = G.', array transpose.
% C = G.' is the array transpose of G.
%
% See also GrB.trans, GrB/ctranspose.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

ghb = 1 ;     % 0 for GrB, 1 for GhB

C = GrB (gbmex_trans (ghb, G)) ;

