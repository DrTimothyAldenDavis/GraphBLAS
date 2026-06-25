function C = asinh (G)
%ASINH inverse hyperbolic sine.
% C = asinh (G) is the inverse hyberbolic sine of each entry G.
%
% See also GrB/sin, GrB/asin, GrB/sinh.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

ghb = 0 ;     % 0 for GrB, 1 for GhB

if (~gb_isfloat (gbmex_type (G)))
    op = 'asinh.double' ;
else
    op = 'asinh' ;
end

C = gzb_apply (ghb, op, G) ;

