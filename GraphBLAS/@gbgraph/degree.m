function d = degree (G)
%DEGREE degree of a GraphBLAS gbgraph
% d = degree (G) is a vector of size n for an n-by-n undirected GraphBLAS graph
% G, equal to d = sum (spones (G)).
%
% See also graph/degree, gbgraph/indegree, gbgraph/outdegree.

% TODO: tests

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

if (isdirected (G))
    error ('G must be an undirected graph') ;
end

if (gb.isbyrow (G))
    d = gb.rowdegree (G) ;
else
    d = gb.coldegree (G) ;
end

