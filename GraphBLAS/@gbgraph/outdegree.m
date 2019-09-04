function d = outdegree (G)
%OUTDEGREE out-degree of a directed or undirected gbgraph.
% d = outdegree (G) is a vector of size n for gbgraph with n nodes equal to d =
% sum (spones (G')).  d(i) is the number of entries in G(i,:).
%
% See also digraph/outdegree, gbgraph/degree, gbgraph/indegree.

% TODO tests

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

if (isundirected (G))
    d = degree (G) ;
else
    d = gb.rowdegree (G) ;
end

