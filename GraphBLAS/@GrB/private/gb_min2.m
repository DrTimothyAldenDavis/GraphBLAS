function C = gb_min2 (op, A, B)
%GB_MIN2 2-input min
% Implements C = min (A,B)

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

[am, an, atype] = gbmex_size (A) ;
[bm, bn, btype] = gbmex_size (B) ;
a_is_scalar = (am == 1) && (an == 1) ;
b_is_scalar = (bm == 1) && (bn == 1) ;
ctype = gbmex_optype (atype, btype) ;

if (a_is_scalar)
    if (b_is_scalar)
        % both A and B are scalars.  Result is also a scalar.
        C = GrB (gbmex_eunion (op, A, 0, B, 0)) ;
    else
        % A is a scalar, B is a matrix
        if (gb_scalar (A) < 0)
            % since A < 0, the result is full
            a = gb_scalar_to_full (bm, bn, ctype, gb_fmt (B), A) ;
            C = GrB (gbmex_eadd (a, op, B)) ;
        else
            % since A >= 0, the result is sparse.
            a = GrB (gbmex_full (A)) ;
            C = GrB (gbmex_apply2 (a, op, B)) ;
        end
    end
else
    if (b_is_scalar)
        % A is a matrix, B is a scalar
        if (gb_scalar (B) < 0)
            % since B < 0, the result is full
            b = gb_scalar_to_full (am, an, ctype, gb_fmt (A), B) ;
            C = GrB (gbmex_eadd (A, op, b)) ;
        else
            % since B >= 0, the result is sparse.
            b = GrB (gbmex_full (B)) ;
            C = GrB (gbmex_apply2 (A, op, b)) ;
        end
    else
        % both A and B are matrices.  Result is sparse.
        C = GrB (gbmex_eunion (op, A, 0, B, 0)) ;
    end
end

