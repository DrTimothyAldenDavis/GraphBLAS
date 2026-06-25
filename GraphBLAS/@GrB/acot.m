function C = acot (G)
%ACOT inverse cotangent.
% C = acot (G) is the inverse cotangent of each entry of G.  Since acot (0)
% is nonzero, C is a full matrix.
%
% See also GrB/cot, GrB/coth, GrB/acoth.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

ghb = 0 ;     % 0 for GrB, 1 for GhB

type = gbmex_type (G) ;
if (~gb_isfloat (type))
    type = 'double' ;
end

C = gzb_apply (ghb, 'atan', gzb_apply (ghb, 'minv', gzb_full (ghb, G, type))) ;

