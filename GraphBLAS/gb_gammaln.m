function C = gb_gammaln (ghb, G)
%GB_GAMMALN implements GrB/gammaln and GhB/gammaln.  Not user-callable.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

type = gbmex_type (G) ;
if (gb_contains (type, 'complex'))
    error ('GrB:error', 'input must be real') ;
end
if (~gb_isfloat (type))
    type = 'double' ;
end

C = gzb_apply (ghb, 'gammaln', gzb_full (ghb, G, type)) ;

