function C = isnan (G)
%ISNAN true for NaN elements.
% C = isnan (G) is a logical C matrix with C(i,j)=true if G(i,j) is NaN.
%
% See also GrB/isinf, GrB/isfinite.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

ghb = 1 ;     % 0 for GrB, 1 for GhB

[m, n, type] = gbmex_size (G) ;

if (gb_isfloat (type))
    C = GrB (gbmex_apply (ghb, 'isnan', G)) ;
else
    % C is all false
    C = GrB (m, n, 'logical') ;
end

