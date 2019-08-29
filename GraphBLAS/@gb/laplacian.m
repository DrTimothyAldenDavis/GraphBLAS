function L = laplacian (G, type)
%LAPLACIAN Graph Laplacian matrix
% L = laplacian (G) is the graph Laplacian matrix of an n-by-n GraphBLAS
% matrix G.  G must be square. The diagonal of G is ignored.  If G is
% unsymmetric, the Laplacian of G+G' is returned.  The diagonal of L is the
% degree of the nodes.  That is, L(j,j) = sum (spones (G (:,j))).  L(i,j) =
% L(j,i) = -1 if the edge (i,j) exists in G.  The type of L defaults to
% double.  With a second argument, the type of L can be specified (see help
% gb.type), as L = laplacian (G,type).
%
% See also graph/laplacian.

% TODO: tests

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

[m n] = size (G) ;
if (m ~= n)
    error ('G must be square') ;
end
if (nargin < 2)
    type = 'double' ;
end
G = spones (G, type) ;
if (~issymmetric (G))
    % make G symmetric
    G = spones (G + G') ;
end
if (any (diag (G)))
    % remove any diagonal entries
    G = gb.select ('offdiag', G) ;
end

D = gb.vreduce ('+', G, struct ('in0', 'transpose')) ;
L = diag (D) - tril (G) - triu (G) ;

