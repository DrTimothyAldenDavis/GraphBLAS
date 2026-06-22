function C = asec (G)
%ASEC inverse secant.
% C = asec (G) is the inverse secant of each entry of G.  Since asec (0) is
% nonzero, the result is a full matrix.  C is complex if any (abs(G) < 1).
%
% See also GrB/sec, GrB/sech, GrB/asech.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

ghb = 1 ;     % 0 for GrB, 1 for GhB

type = gbmex_type (G) ;
if (~gb_isfloat (type))
    type = 'double' ;
end

C = gb_trig ('acos', GrB (gbmex_apply (ghb, 'minv', GrB (gbmex_full (ghb, G, type))))) ;

