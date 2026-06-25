function C = sinh (G)
%SINH hyperbolic sine.
% C = sinh (G) is the hyperbolic sine of each entry of G.
%
% See also GrB/sin, GrB/asin, GrB/asinh.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

ghb = 0 ;     % 0 for GrB, 1 for GhB

if (~gb_isfloat (gbmex_type (G)))
    op = 'sinh.double' ;
else
    op = 'sinh' ;
end

C = GrB (gbmex_apply (ghb, op, G)) ;

