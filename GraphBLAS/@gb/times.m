function C = times (A, B)
%TIMES C = A.*B, sparse matrix element-wise multiplication
% If both A and B are matrices, the pattern of C is the intersection of A
% and B.  If one is a scalar, the pattern of C is the same as the pattern
% of the one matrix.
%
% See also gb.emult.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

if (isscalar (A))
    if (isscalar (B))
        % both A and B are scalars
        C = gb.emult ('*', A, B) ;
    else
        % A is a scalar, B is a matrix
        C = gb.emult ('*', gb.expand (A, B), B) ;
    end
else
    if (isscalar (B))
        % A is a matrix, B is a scalar
        C = gb.emult ('*', A, gb.expand (B, A)) ;
    else
        % both A and B are matrices
        C = gb.emult ('*', A, B) ;
    end
end

