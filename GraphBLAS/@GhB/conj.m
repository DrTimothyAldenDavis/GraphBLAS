function C = conj (G)
%CONJ complex conjugate.
% C = conj (G) is the complex conjugate of each entry of G.
%
% See also GrB/real, GrB/imag.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

ghb = 1 ;     % 0 for GrB, 1 for GhB

C = gb_conj (ghb, G) ;

