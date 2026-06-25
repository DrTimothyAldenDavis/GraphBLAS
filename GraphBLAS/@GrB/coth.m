function C = coth (G)
%COTH hyperbolic cotangent.
% C = coth (G) is the hyperbolic cotangent of each entry of G.  Since coth
% (0) is nonzero, C is a full matrix.
%
% See also GrB/cot, GrB/acot, GrB/acoth.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

ghb = 0 ;     % 0 for GrB, 1 for GhB

if (~gb_isfloat (gbmex_type (G)))
    op = 'tanh.double' ;
else
    op = 'tanh' ;
end

C = gzb_apply (ghb, 'minv', gzb_full (ghb, gzb_apply (ghb, op, G))) ;

