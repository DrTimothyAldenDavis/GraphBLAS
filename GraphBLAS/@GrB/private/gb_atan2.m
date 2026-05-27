function C = gb_atan2 (A, B)
%GB_ATAN2 four quadrant inverse tangent.
% C = atan2 (X,Y) is the 4 quadrant arctangent of the entries in X and Y.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

% atan2(A,B) gives the set union of the pattern of A and B

if (gb_isscalar (A))
    if (gb_isscalar (B))
        % both A and B are scalars
        C = GrB (gbemult ('atan2', A, B)) ;
    else
        % A is a scalar, B is a matrix
        C = GrB (gbapply2 ('atan2', gbfull (A), B)) ;
    end
else
    if (gb_isscalar (B))
        % A is a matrix, B is a scalar
        C = GrB (gbapply2 ('atan2', A, gbfull (B))) ;
    else
        % both A and B are matrices.  C is the set union of A and B.
        C = GrB (gbeunion ('atan2', A, 0, B, 0)) ;
    end
end

