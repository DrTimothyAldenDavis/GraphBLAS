function s = gb_make_real (G)
%GB_MAKE_REAL true if a complex matrix G has all-zero imag(G)

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

ghb = 1 ;     % 0 for GrB, 1 for GhB

s = gb_contains (gbmex_type (G), 'complex') && ...
    (gbmex_nvals (GrB (gbmex_select (ghb, 'nonzero', GrB (gbmex_apply (ghb, 'cimag', G))))) == 0) ;

