function C = tanh (G)
%TANH hyperbolic tangent.
% C = tanh (G) is the hyperbolic tangent of each entry of G.
%
% See also GrB/tan, GrB/atan, GrB/atanh, GrB/atan2.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

ghb = 0 ;     % 0 for GrB, 1 for GhB

if (~gb_isfloat (gbmex_type (G)))
    op = 'tanh.double' ;
else
    op = 'tanh' ;
end

C = GrB (gbmex_apply (ghb, op, G)) ;

