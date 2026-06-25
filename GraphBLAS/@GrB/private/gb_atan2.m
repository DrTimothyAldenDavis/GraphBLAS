function C = gb_atan2 (A, B)
%GB_ATAN2 four quadrant inverse tangent.
% C = atan2 (X,Y) is the 4 quadrant arctangent of the entries in X and Y.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

ghb = 1 ;     % 0 for GrB, 1 for GhB

% atan2(A,B) gives the set union of the pattern of A and B

if (gb_isscalar (A))
    if (gb_isscalar (B))
        % both A and B are scalars
        C = GrB (gbmex_emult (ghb, 'atan2', A, B)) ;
    else
        % A is a scalar, B is a matrix
        a = GrB (gbmex_full (ghb, A)) ;
        C = GrB (gbmex_apply2 (ghb, 'atan2', a, B)) ;
    end
else
    if (gb_isscalar (B))
        % A is a matrix, B is a scalar
        b = GrB (gbmex_full (ghb, B)) ;
        C = GrB (gbmex_apply2 (ghb, 'atan2', A, b)) ;
    else
        % both A and B are matrices.  C is the set union of A and B.
        C = GrB (gbmex_eunion (ghb, 'atan2', A, 0, B, 0)) ;
    end
end

