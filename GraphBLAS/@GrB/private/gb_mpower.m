function C = gb_mpower (A, b)
%GB_MPOWER C = A^b where b > 0 is an integer

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (b == 1)
    C = GrB (A) ;
else
    C = gb_mpower (A, floor (b/2)) ;
    C = GrB (gbmex_mxm (C, '+.*', C)) ;
    if (mod (b, 2) == 1)
        C = GrB (gbmex_mxm (C, '+.*', A)) ;
    end
end

