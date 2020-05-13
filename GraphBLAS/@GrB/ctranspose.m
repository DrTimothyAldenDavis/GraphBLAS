function C = ctranspose (G)
%CTRANSPOSE C = G', transpose a GraphBLAS matrix.
%
% See also GrB.trans, GrB/transpose, GrB/conj.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights
% Reserved. http://suitesparse.com.  See GraphBLAS/Doc/License.txt.

if (isreal (G))
    C = GrB.trans (G) ;
else
    C = GrB.trans (conj (G)) ;
end

