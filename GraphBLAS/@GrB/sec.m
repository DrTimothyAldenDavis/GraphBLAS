function C = sec (G)
%SEC secant.
% C = sec (G) is the secant of each entry of G.
% Since sec (0) = 1, the result is a full matrix.
%
% See also GrB/asec, GrB/sech, GrB/asech.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

ghb = 0 ;     % 0 for GrB, 1 for GhB

type = gbmex_type (G) ;
if (~gb_isfloat (type))
    type = 'double' ;
end

C = gzb_apply (ghb, 'minv', gzb_apply (ghb, 'cos', gzb_full (ghb, G, type))) ;

