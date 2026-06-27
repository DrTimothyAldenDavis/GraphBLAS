function s = gb_make_real (ghb, G)
%GB_MAKE_REAL true if a complex matrix has zero imag part.  Not user-callable.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

s = gb_contains (gbmex_type (G), 'complex') && ...
    (gbmex_nvals (gzb_select (ghb, 'nonzero', gzb_apply (ghb, 'cimag', G))) == 0) ;

