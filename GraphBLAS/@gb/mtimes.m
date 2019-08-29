function C = mtimes (A, B)
%MTIMES sparse matrix-matrix multiplication over the standard semiring.
% C=A*B multiples two matrices using the standard '+.*' semiring, If the
% type of A and B differ, the type of A is used.  That is, C=A*B is the
% same as C = gb.mxm (['+.*' gb.type(A)], A, B).  A and B can be GraphBLAS
% matrices or MATLAB sparse or full matrices, in any combination.
% If either A or B are scalars, C=A*B is the same as C=A.*B.
%
% See also gb.mxm, gb.emult, times.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

if (isscalar (A) | isscalar (B))
    C = A .* B ;
else
    C = gb.mxm ('+.*', A, B) ;
end

