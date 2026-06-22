function C = rdivide (A_arg, B)
%RDIVIDE C = A./B, sparse matrix element-wise division.
% C = A./B when B is a matrix results in a full matrix C, with all
% entries present.  If A is a matrix and B is a scalar, then C has the
% pattern of A, except if B is zero and A is double, single, or complex.
% In that case, since 0/0 is NaN, C is a full matrix.
%
% See also GrB/ldivide, GrB.emult, GrB.eadd.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

ghb = 1 ;     % 0 for GrB, 1 for GhB

[am, an, atype] = gbmex_size (A_arg) ;
[bm, bn, btype] = gbmex_size (B) ;
a_is_scalar = (am == 1) && (an == 1) ;
b_is_scalar = (bm == 1) && (bn == 1) ;
ctype = gbmex_optype (atype, btype) ;

if (a_is_scalar && gb_scalar (A_arg) == 0 && gb_isfloat (ctype))
    A = 0 ;
else
    A = A_arg ;
end

if (a_is_scalar)
    if (b_is_scalar)
        % both A and B are scalars
        b = GrB (gbmex_full (ghb, B)) ;
        C = GrB (gbmex_emult (ghb, A, '/', b)) ;
    else
        % A is a scalar, B is a matrix.
        % Expand B to full with type of C
        b = GrB (gbmex_full (ghb, B, ctype)) ;
        C = GrB (gbmex_apply2 (ghb, A, '/', b)) ;
    end
else
    if (b_is_scalar)
        % A is a matrix, B is a scalar
        if (gb_scalar (B) == 0 && gb_isfloat (atype))
            % 0/0 is Nan, and thus must be computed computed if A is
            % floating-point.  The result is a full matrix.
            % expand B into a full matrix and cast to the type of A
            b = gb_scalar_to_full (am, an, atype, gb_fmt (A), B) ;
            C = GrB (gbmex_emult (ghb, A, '/', b)) ;
        else
            % The scalar B is nonzero so just compute A/B in the pattern
            % of A.  The result is sparse (the pattern of A).
            C = GrB (gbmex_apply2 (ghb, A, '/', B)) ;
        end
    else
        % both A and B are matrices.  The result is a full matrix.
        a = GrB (gbmex_full (ghb, A, ctype)) ;
        b = GrB (gbmex_full (ghb, B, ctype)) ;
        C = GrB (gbmex_emult (ghb, a, '/', b)) ;
    end
end

