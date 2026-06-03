function C = gb_make_real (G)
%GB_MAKE_REAL convert complex matrix to real if imag(G) is zero

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (gb_contains (gbtype (G), 'complex'))
    ci = GrB (gbapply ('cimag', G)) ;
    s = GrB (gbselect ('nonzero', ci)) ;
    if (gbnvals (s) == 0)
        C = GrB (gbapply ('creal', G)) ;
    else
        C = GrB (G) ;
    end
else
    C = GrB (G) ;
end

