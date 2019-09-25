function C = or (A, B)
%| logical OR.
% C = (A | B) is the element-by-element logical OR of A and B.  One or
% both may be scalars.  Otherwise, A and B must have the same size.
% GraphBLAS and MATLAB matrices may be combined.
%
% See also gb/and, gb/xor.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

if (isscalar (A))
    if (isscalar (B))
        % A and B are scalars
        C = gb.prune (gb.emult ('|.logical', A, B)) ;
    else
        % A is a scalar, B is a matrix
        if (gb_get_scalar (A) == 0)
            % A is false, so C is B typecasted to logical
            C = gb (gb.prune (B), 'logical') ;
        else
            % A is true, so C is a full matrix the same size as B
            C = gb (true (size (B))) ;
        end
    end
else
    if (isscalar (B))
        % A is a matrix, B is a scalar
        if (gb_get_scalar (B) == 0)
            % B is false, so C is A typecasted to logical
            C = gb (gb.prune (A), 'logical') ;
        else
            % B is true, so C is a full matrix the same size as A
            C = gb (true (size (A))) ;
        end
    else
        % both A and B are matrices.  C is the set union of A and B
        C = gb.prune (gb.eadd ('|.logical', A, B)) ;
    end
end

