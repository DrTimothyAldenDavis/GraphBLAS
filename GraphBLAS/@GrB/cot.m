function C = cot (G)
%COT cotangent.
% C = cot (G) is the cotangent of each entry of G.  Since cot (0) is
% nonzero, C is a full matrix.
%
% See also GrB/coth, GrB/acot, GrB/acoth.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

ghb = 1 ;     % 0 for GrB, 1 for GhB

if (~gb_isfloat (gbmex_type (G)))
    op = 'tan.double' ;
else
    op = 'tan' ;
end

C = GrB (gbmex_apply (ghb, 'minv', GrB (gbmex_full (ghb, GrB (gbmex_apply (ghb, op, G)))))) ;

