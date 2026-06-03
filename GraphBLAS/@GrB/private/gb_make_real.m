function s = gb_make_real (G)
%GB_MAKE_REAL true if a complex matrix G has all-zero imag(G)

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

s = gb_contains (gbtype (G), 'complex') && ...
    (gbnvals (GrB (gbselect ('nonzero', GrB (gbapply ('cimag', G))))) == 0) ;

