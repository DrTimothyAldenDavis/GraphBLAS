function C = gb_double (ghb, G)
%GB_DOUBLE implements GrB/double and GhB/double.  Not user-callable.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (gb_contains (gbmex_type (G), 'complex'))
    type = 'double complex' ;
else
    type = 'double' ;
end

C = gbmex_builtin (gzb_cast (ghb, G, type)) ;

