function C = gb_eadd (A, op, B)
%GB_EADD C = A+B, sparse matrix 'addition' using the given op.
% The pattern of C is the set union of A and B.  This method assumes the
% identity value of the op is zero.  That is, x+0 = x+0 = x.  The binary
% operator op is only applied to entries in the intersection of the
% pattern of A and B.
%
% See also GrB/plus, GrB/minus, GrB/bitxor, GrB/bitor, GrB/hypot.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

if (isscalar (A))
    if (isscalar (B))
        % both A and B are scalars.  Result is also a scalar.
        C = GrB.eadd (A, op, B) ;
    else
        % A is a scalar, B is a matrix.  Result is full, unless A == 0.
        if (gb_get_scalar (A) == 0)
            C = B ;
        else
            % A (1:m,1:n) = A and cast to ctype
            ctype = GrB.optype (A, B) ;
            [m, n] = size (B) ;
            A = GrB.subassign (GrB (m, n, ctype), A) ;
            C = GrB.eadd (A, op, B) ;
        end
    end
else
    if (isscalar (B))
        % A is a matrix, B is a scalar.  Result is full.
        C = gb_eadd (B, op, A) ;
    else
        % both A and B are matrices.  Result is sparse.
        C = GrB.eadd (A, op, B) ;
    end
end

