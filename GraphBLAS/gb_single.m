function C = gb_single (ghb, G)
%GB_SINGLE implements GrB/single and GhB/single.  Not user-callable.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

desc.kind = 'builtin' ;
if (gb_contains (gbmex_type (G), 'complex'))
    z = complex (single (0)) ;
    ctype = 'single complex' ;
else
    z = single (0) ;
    ctype = 'single' ;
end

% export C as a full matrix
C = gbmex_builtin (gzb_full (ghb, G, ctype, z, desc)) ;

