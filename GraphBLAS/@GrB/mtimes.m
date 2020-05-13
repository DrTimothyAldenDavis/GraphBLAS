function C = mtimes (A, B)
%MTIMES sparse matrix-matrix multiplication over the standard semiring.
% C=A*B multiples two matrices using the standard '+.*' semiring.  If
% either A or B are scalars, C=A*B is the same as C=A.*B.
%
% The input matrices may be either GraphBLAS and/or MATLAB matrices, in
% any combination.  C is returned as a GraphBLAS matrix.
%
% See also GrB.mxm, GrB.emult, GrB/times.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights
% Reserved. http://suitesparse.com.  See GraphBLAS/Doc/License.txt.

if (isscalar (A) || isscalar (B))
    C = A .* B ;
else
    C = GrB.mxm (A, '+.*', B) ;
end

