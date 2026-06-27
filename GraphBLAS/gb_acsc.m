function C = gb_acsc (ghb, G)
%GB_ACSC implements GrB/acsc and GhB/acsc.  Not user-callable.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

type = gbmex_type (G) ;
if (~gb_isfloat (type))
    type = 'double' ;
end

C = gb_trig (ghb, 'asin', gzb_apply (ghb, 'minv', gzb_full (ghb, G, type))) ;

