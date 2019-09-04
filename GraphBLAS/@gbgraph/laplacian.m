function L = laplacian (G, type)
%LAPLACIAN Graph Laplacian matrix
% L = laplacian (G) is the graph Laplacian of the gbgraph G.  If G is a
% directed graph, the Laplacian of G+G' is computed.  The diagonal of L is the
% degree of the nodes.  Self-edges are ignored.  Assuming G has no self-edges,
% L(j,j) = sum (spones (G (:,j))).  For off-diagonal entries, L(i,j) = L(j,i) =
% -1 if the edge (i,j) exists in G.
%
% The type of L defaults to double.  With a second argument, the type of L can
% be specified, as L = laplacian (G,type); type may be 'double', 'single',
% 'int8', 'int16', 'int32', or 'int64'.  Be aware that integer overflow may
% occur with the smaller integer types.
%
% L is returned as an undirected gbgraph.
%
% See also graph/laplacian.

% TODO: tests

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

if (nargin < 2)
    type = 'double' ;
end
G_is_directed = isdirected (G) ;

if (~ (isequal (type, 'double') || isequal (type, 'single') || ...
       isequal (type (1:3), 'int')))
    % type must be 'double', 'single', 'int8', 'int16', 'int32', or 'int64'.
    error ('invalid type') ;
end

G = spones (G, type) ;
if (G_is_directed)
    % make G symmetric
    G = spones (G + G') ;
end
if (any (diag (G)))
    % remove any diagonal entries
    G = gb.select ('offdiag', G) ;
end

% construct the Laplacian
D = gb.vreduce ('+', G, struct ('in0', 'transpose')) ;
L = gbgraph (diag (D) - tril (G) - triu (G), 'undirected', false) ;

