function C = gb_erfc (ghb, G)
%GB_ERFC implements GrB/erfc and GhB/erfc.  Not user-callable.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

type = gbmex_type (G) ;
if (gb_contains (type, 'complex'))
    error ('GrB:error', 'input must be real') ;
end
if (~gb_isfloat (type))
    type = 'double' ;
end

C = gzb_apply (ghb, 'erfc', gzb_full (ghb, G, type)) ;

