function C = isinf (G)
%ISINF true for infinite elements.
% C = isinf (G) returns a logical matrix C where C(i,j) = true
% if G(i,j) is infinite.
%
% See also GrB/isnan, GrB/isfinite.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

ghb = 0 ;     % 0 for GrB, 1 for GhB

[m, n, type] = gbmex_size (G) ;

if (gb_isfloat (type))
    C = GrB (gbmex_apply (ghb, 'isinf', G)) ;
else
    % C is all false
    C = GrB (m, n, 'logical') ;
end

