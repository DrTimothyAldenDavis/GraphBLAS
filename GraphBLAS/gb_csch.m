function C = gb_csch (ghb, G)
%GB_CSCH implements GrB/csch and GhB/csch.  Not user-callable.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (~gb_isfloat (gbmex_type (G)))
    op = 'sinh.double' ;
else
    op = 'sinh' ;
end

C = gzb_apply (ghb, 'minv', gzb_full (ghb, gzb_apply (ghb, op, G))) ;

