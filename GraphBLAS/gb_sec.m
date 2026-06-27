function C = gb_sec (ghb, G)
%GB_SEC implements GrB/sec and GhB/sec.  Not user-callable.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

type = gbmex_type (G) ;
if (~gb_isfloat (type))
    type = 'double' ;
end

C = gzb_apply (ghb, 'minv', gzb_apply (ghb, 'cos', gzb_full (ghb, G, type))) ;

