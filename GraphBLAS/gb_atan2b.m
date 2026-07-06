function C = gb_atan2b (ghb, A, B)
%GB_ATAN2B four quadrant inverse tangent.  Not user-callable.
% C = atan2b (X,Y) is the 4 quadrant arctangent of the entries in X and Y.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

% atan2b(A,B) gives the set union of the pattern of A and B.

if (gb_is_grb (A))
    A = struct (A) ;
end

if (gb_is_grb (B))
    B = struct (B) ;
end

if (gb_isscalar (A))
    if (gb_isscalar (B))
        % both A and B are scalars
        C = gzb_emult (ghb, 'atan2', A, B) ;
    else
        % A is a scalar, B is a matrix
        a = gzb_full (ghb, A) ;
        C = gzb_apply2 (ghb, 'atan2', a, B) ;
    end
else
    if (gb_isscalar (B))
        % A is a matrix, B is a scalar
        b = gzb_full (ghb, B) ;
        C = gzb_apply2 (ghb, 'atan2', A, b) ;
    else
        % both A and B are matrices.  C is the set union of A and B.
        C = gzb_eunion (ghb, 'atan2', A, 0, B, 0) ;
    end
end

