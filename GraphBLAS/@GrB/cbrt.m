function C = cbrt (G)
%CBRT cube root
% C = cbrt (G) is the cube root of the entries of G.
%
% See also GrB/sqrt, nthroot.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

ghb = 0 ;     % 0 for GrB, 1 for GhB

C = gb_cbrt (ghb, G) ;

