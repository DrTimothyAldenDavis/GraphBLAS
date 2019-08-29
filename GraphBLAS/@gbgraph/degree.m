function d = degree (H)
%DEGREE degree of a GraphBLAS gbgraph
% d = degree (H) is a vector of size n for an n-by-n undirected GraphBLAS graph
% H, equal to the d = sum (spones (H)).
%
% See also graph/degree, gb/indegree, gb/outdegree.

% TODO: tests

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

if (~isequal (H.graphkind, 'graph'))
    error ('H must be an undirected graph') ;
end

if (isequal (gb.format (G), 'by row'))
    d = row_degree (G) ;
else
    d = col_degree (G) ;
end

