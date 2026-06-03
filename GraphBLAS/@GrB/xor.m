function C = xor (A, B)
%XOR logical exclusive OR.
% C = xor (A,B) is the element-by-element logical OR of A and B.  One or
% both may be scalars.  Otherwise, A and B must have the same size.
%
% See also GrB/and, GrB/or, GrB/not.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (gb_isscalar (A))
    if (gb_isscalar (B))
        % A and B are scalars
        C = GrB (gbemult (A, 'xor.logical', B)) ;
    else
        % A is a scalar, B is a matrix
        if (gb_scalar (A) == 0)
            % A is false, so C is B typecasted to logical
            C = GrB (B, 'logical') ;
        else
            % A is true, so C is a full matrix the same size as B
            b = GrB (gbfull (B, 'logical')) ;
            C = GrB (gbapply ('~', b)) ;
        end
    end
else
    if (gb_isscalar (B))
        % A is a matrix, B is a scalar
        if (gb_scalar (B) == 0)
            % B is false, so C is A typecasted to logical
            C = GrB (A, 'logical') ;
        else
            % B is true, so C is a full matrix the same size as A
            a = GrB (gbfull (A, 'logical')) ;
            C = GrB (gbapply ('~', a)) ;
        end
    else
        % both A and B are matrices.  C is the set union of A and B
        C = GrB (gbeadd (A, 'xor.logical', B)) ;
    end
end

