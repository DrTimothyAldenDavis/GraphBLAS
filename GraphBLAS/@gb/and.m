function C = and (A, B)
%& logical AND.
% Element-by-element logical AND of A and B.  One or both may be scalars.
% Otherwise, A and B must have the same size.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

if (isscalar (A))
    if (isscalar (B))
        % A and B are scalars
        C = gb.select ('nonzero', gb.emult ('&.logical', A, B)) ;
    else
        % A is a scalar, B is a matrix
        if (get_scalar (A) == 0)
            % A is false, so C is empty, the same size as B
            [m n] = size (B) ;
            C = gb (m, n, 'logical') ;
        else
            % A is true, so C is B typecasted to logical
            C = gb (gb.select ('nonzero', B), 'logical') ;
        end
    end
else
    if (isscalar (B))
        % A is a matrix, B is a scalar
        if (get_scalar (B) == 0)
            % B is false, so C is empty, the same size as A
            [m n] = size (A) ;
            C = gb (m, n, 'logical') ;
        else
            % B is true, so C is A typecasted to logical
            C = gb (gb.select ('nonzero', A), 'logical') ;
        end
    else
        % both A and B are matrices.  C is the set intersection of A and B
        C = gb.select ('nonzero', gb.emult ('&.logical', A, B)) ;
    end
end

