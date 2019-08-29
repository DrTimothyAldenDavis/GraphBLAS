function L = laplacian (H, type)
%LAPLACIAN Graph Laplacian matrix
% L = laplacian (H) is the graph Laplacian matrix of an n-by-n GraphBLAS
% gbgraph H.  If H is a directed graph, the Laplacian of H+H' is computed.  The
% diagonal of L is the degree of the nodes, including self-edges if present.
% That is, L(j,j) = sum (spones (H (:,j))).  L(i,j) = L(j,i) = -1 if the edge
% (i,j) exists in G.  The type of L defaults to double.  With a second
% argument, the type of L can be specified, as L = laplacian (H,type);
% type may be 'double', 'single', 'int8', 'int16', 'int32', or 'int64'.
% Be aware that integer overflow may occur with the smaller integer types.
%
% See also graph/laplacian.

% TODO: tests

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

if (nargin < 2)
    type = 'double' ;
end
kind = H.graphkind ;

ok = isequal (type, 'double') || isequal (type, 'single') || ...
     isequal (type (1:3), 'int') ;
if (~ok)
    % type must be 'double', 'single', 'int8', 'int16', 'int32', or 'int64'.
    error ('invalid type') ;
end

H = spones (H, type) ;
if (isequal (kind, 'digraph'))
    % make H symmetric
    H = spones (H + H') ;
end
if (any (diag (H)))
    % remove any diagonal entries
    H = gb.select ('offdiag', H) ;
end

D = gb.vreduce ('+', H, struct ('in0', 'transpose')) ;
L = diag (D) - tril (H) - triu (H) ;

