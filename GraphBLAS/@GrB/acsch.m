function C = acsch (G)
%ACSCH inverse hyperbolic cosecant.
% C = acsch (G) is the inverse hyberbolic cosecant of each entry G.  Since
% acsch (0) is nonzero, C is a full matrix.
%
% See also GrB/csc, GrB/acsc, GrB/csch.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

ghb = 0 ;     % 0 for GrB, 1 for GhB

type = gbmex_type (G) ;
if (~gb_isfloat (type))
    type = 'double' ;
end

C = GrB (gbmex_apply (ghb, 'asinh', GrB (gbmex_apply (ghb, 'minv', GrB (gbmex_full (ghb, G, type)))))) ;

