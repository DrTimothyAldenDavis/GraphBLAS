function C = gb_cosh (ghb, G)
%GB_COSH implements GrB/cosh and GhB/cosh.  Not user-callable.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

type = gbmex_type (G) ;
if (~gb_isfloat (type))
    type = 'double' ;
end

C = gzb_apply (ghb, 'cosh', gzb_full (ghb, G, type)) ;

