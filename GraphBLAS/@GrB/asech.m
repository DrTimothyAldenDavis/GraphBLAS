function C = asech (G)
%ASECH inverse hyperbolic secant.
% C = asech (G) is the inverse hyperbolic secant of each entry of G.  Since
% asech (0) is nonzero, the result is a full matrix.  C is complex if G is
% complex, or if any real entries are outside of the range [0,1].
%
% See also GrB/sec, GrB/asec, GrB/sech.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

type = gbmex_type (G) ;
if (~gb_isfloat (type))
    type = 'double' ;
end

C = gb_trig ('acosh', GrB (gbmex_apply ('minv', GrB (gbmex_full (G, type))))) ;

