function C = gb_sech (ghb, G)
%GB_SECH implements GrB/sech and GhB/sech.  Not user-callable.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

type = gbmex_type (G) ;
if (~gb_isfloat (type))
    type = 'double' ;
end

C = gzb_apply (ghb, 'minv', gzb_apply (ghb, 'cosh', gzb_full (ghb, G, type))) ;

