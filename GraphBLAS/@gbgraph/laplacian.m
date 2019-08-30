function L = laplacian (H, type)
%LAPLACIAN Graph Laplacian matrix
% L = laplacian (H) is the graph Laplacian of the gbgraph H.  If H is a
% directed graph, the Laplacian of H+H' is computed.  The diagonal of L is the
% degree of the nodes.  Self-edges are ignored.  Assuming H has no self-edges,
% L(j,j) = sum (spones (H (:,j))).  For off-diagonal entries, L(i,j) = L(j,i) =
% -1 if the edge (i,j) exists in H.
%
% The type of L defaults to double.  With a second argument, the type of L can
% be specified, as L = laplacian (H,type); type may be 'double', 'single',
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
kind = H.graphkind ;

if (~ (isequal (type, 'double') || isequal (type, 'single') || ...
       isequal (type (1:3), 'int')))
    % type must be 'double', 'single', 'int8', 'int16', 'int32', or 'int64'.
    error ('invalid type') ;
end

H = spones (H, type) ;
if (isequal (kind, 'directed'))
    % make H symmetric
    H = spones (H + H') ;
end
if (any (diag (H)))
    % remove any diagonal entries
    H = gb.select ('offdiag', H) ;
end

% construct the Laplacian
D = gb.vreduce ('+', H, struct ('in0', 'transpose')) ;
L = gbgraph (diag (D) - tril (H) - triu (H), 'undirected', false) ;

