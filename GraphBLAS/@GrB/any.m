function C = any (G, option)
%ANY True if any element of a GraphBLAS matrix is nonzero or true.
%
% C = any (G) is true if any entry in G is nonzero or true.  If G is a
% matrix, C is a row vector with C(j) = any (G (:,j)).
%
% C = any (G, 'all') is a scalar, true if any entry in G is nonzero or true
% C = any (G, 1) is a row vector with C(j) = any (G (:,j))
% C = any (G, 2) is a column vector with C(i) = any (G (i,:))
%
% See also GrB/all, GrB/nnz, GrB.entries, GrB.nonz.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights
% Reserved. http://suitesparse.com.  See GraphBLAS/Doc/License.txt.

if (nargin == 1)
    option = 'default' ;
end

C = GrB (gb_any (G.opaque, option)) ;

