function C = gb_mpower (A, b)
%GB_MPOWER C = A^b where b > 0 is an integer

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

ghb = 0 ;     % 0 for GrB, 1 for GhB

if (b == 1)
    C = gzb (ghb, A) ;
else
    C = gb_mpower (A, floor (b/2)) ;
    C = gzb_mtimes (ghb, C, C) ;
    if (mod (b, 2) == 1)
        C = gzb_mtimes (ghb, C, A) ;
    end
end

