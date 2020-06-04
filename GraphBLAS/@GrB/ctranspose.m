function C = ctranspose (G)
%CTRANSPOSE C = G', transpose a GraphBLAS matrix.
%
% See also GrB.trans, GrB/transpose, GrB/conj.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights
% Reserved. http://suitesparse.com.  See GraphBLAS/Doc/License.txt.

G = G.opaque ;

if (contains (gbtype (G), 'complex'))
    C = GrB (gbtrans (gbapply ('conj', G))) ;
else
    C = GrB (gbtrans (G)) ;
end

