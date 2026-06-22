function C = sec (G)
%SEC secant.
% C = sec (G) is the secant of each entry of G.
% Since sec (0) = 1, the result is a full matrix.
%
% See also GrB/asec, GrB/sech, GrB/asech.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

ghb = 1 ;     % 0 for GrB, 1 for GhB

type = gbmex_type (G) ;
if (~gb_isfloat (type))
    type = 'double' ;
end

C = GrB (gbmex_apply (ghb, 'minv', GrB (gbmex_apply (ghb, 'cos', GrB (gbmex_full (ghb, G, type)))))) ;

