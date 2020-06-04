function C = gb_emult (A, op, B)
%GB_EMULT C = A.*B, sparse matrix element-wise multiplication.
% C = gb_emult (A, op, B) computes the element-wise multiplication of A
% and B using the operator op, where the op is '*' for C=A.*B.  If both A
% and B are matrices, the pattern of C is the intersection of A and B.  If
% one is a scalar, the pattern of C is the same as the pattern of the one
% matrix.
%
% The input matrices may be either GraphBLAS structs and/or MATLAB matrices,
% in any combination.  C is returned as a GraphBLAS struct.
%
% See also GrB/times, GrB/bitand, GrB.emult.

% SuiteSparse:GraphBLAS, T. A. Davis, (c) 2017-2020, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

if (gb_isscalar (A))
    if (gb_isscalar (B))
        % both A and B are scalars
    else
        % A is a scalar, B is a matrix
        A = gb_expand (A, B) ;
    end
else
    if (gb_isscalar (B))
        % A is a matrix, B is a scalar
        B = gb_expand (B, A) ;
    else
        % both A and B are matrices
    end
end

C = gbemult (A, op, B) ;

