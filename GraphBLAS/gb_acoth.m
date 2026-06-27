function C = gb_acoth (ghb, G)
%GB_ACOTH implements GrB/acoth and GhB/acoth.  Not user-callable.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

type = gbmex_type (G) ;
if (~gb_isfloat (type))
    type = 'double' ;
end

C = gb_trig (ghb, 'atanh', gzb_apply (ghb, 'minv', gzb_full (ghb, G, type))) ;

