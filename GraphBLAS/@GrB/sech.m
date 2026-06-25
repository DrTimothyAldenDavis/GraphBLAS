function C = sech (G)
%SECH hyperbolic secant.
% C = sech (G) is the hyperbolic secant of each entry of G.
% Since sech(0) is nonzero, C is a full matrix.
%
% See also GrB/sec, GrB/asec, GrB/asech.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

ghb = 0 ;     % 0 for GrB, 1 for GhB

type = gbmex_type (G) ;
if (~gb_isfloat (type))
    type = 'double' ;
end

C = GrB (gbmex_apply (ghb, 'minv', GrB (gbmex_apply (ghb, 'cosh', GrB (gbmex_full (ghb, G, type)))))) ;

