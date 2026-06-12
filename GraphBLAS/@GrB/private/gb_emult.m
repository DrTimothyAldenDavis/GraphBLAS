function C = gb_emult (A, op, B)
%GB_EMULT C = A.*B, sparse matrix element-wise multiplication.
% C = gb_emult (A, op, B) computes the element-wise multiplication of A
% and B using the operator op, where the op is '*' for C=A.*B.  If both A
% and B are matrices, the pattern of C is the intersection of A and B.  If
% one is a scalar, the pattern of C is the same as the pattern of the one
% matrix.
%
% The input matrices may be either GraphBLAS structs and/or built-in
% matrices, in any combination.  C is returned as a GraphBLAS struct.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (gb_isscalar (A))
    if (gb_isscalar (B))
        % both A and B are scalars
        C = GrB (gbmex_emult (A, op, B)) ;
    else
        % A is a scalar, B is a matrix
        C = GrB (gbmex_apply2 (GrB (gbmex_full (A)), op, B)) ;
    end
else
    if (gb_isscalar (B))
        % A is a matrix, B is a scalar
        C = GrB (gbmex_apply2 (A, op, GrB (gbmex_full (B)))) ;
    else
        % both A and B are matrices
        C = GrB (gbmex_emult (A, op, B)) ;
    end
end

