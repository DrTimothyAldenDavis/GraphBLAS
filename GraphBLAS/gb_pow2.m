function C = gb_pow2 (ghb, A, B)
%GB_POW2 implements GrB/pow2 and GhB/pow2.  Not user-callable.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

atype = gbmex_type (A) ;

if (nargin == 2)
    % C = 2.^A
    if (~gb_isfloat (atype))
        atype = 'double' ;
    end
    C = gzb_apply (ghb, 'pow2', gzb_full (ghb, A, atype)) ;
else
    % C = A.*(2.^B)
    type = gbmex_optype (atype, gbmex_type (B)) ;
    if (gb_contains (type, 'single'))
        type = 'single' ;
    else
        type = 'double' ;
    end
    C = gb_eunion (ghb, A, ['pow2.' type], B) ;
end

