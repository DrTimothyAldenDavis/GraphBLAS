function d = degree (H)
%DEGREE degree of a GraphBLAS gbgraph
% d = degree (H) is a vector of size n for an n-by-n undirected GraphBLAS graph
% H, equal to d = sum (spones (H)).
%
% See also graph/degree, gbgraph/indegree, gbgraph/outdegree.

% TODO: tests

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

if (~isequal (H.graphkind, 'undirected'))
    error ('H must be an undirected graph') ;
end

if (isequal (gb.format (H), 'by row'))
    d = gb.rowdegree (H) ;
else
    d = gb.coldegree (H) ;
end

