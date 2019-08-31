function C = plus (A, B)
%PLUS sparse matrix addition, C = A+B.
% C = A+B adds the two matrices A and B.  If A and B are matrices, the
% pattern of C is the set union of A and B.  If one of A or B is a
% scalar, the scalar is expanded into a dense matrix the size of the
% other matrix, and the result is a dense matrix.  If the type of A and B
% differ, the type of A is used, as: C = A + gb (B, gb.type (A)).
%
% The input matrices may be either GraphBLAS and/or MATLAB matrices, in
% any combination.  C is returned as a GraphBLAS matrix.
%
% See also gb.eadd, minus, uminus.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

if (isscalar (A))
    if (isscalar (B))
        % both A and B are scalars.  Result is also a scalar.
        C = gb.eadd ('+', A, B) ;
    else
        % A is a scalar, B is a matrix.  Result is full.
        C = gb.eadd ('+', gb.expand (A, true (size (B))), B) ;
    end
else
    if (isscalar (B))
        % A is a matrix, B is a scalar.  Result is full.
        C = gb.eadd ('+', A, gb.expand (B, true (size (A)))) ;
    else
        % both A and B are matrices.  Result is sparse.
        C = gb.eadd ('+', A, B) ;
    end
end

