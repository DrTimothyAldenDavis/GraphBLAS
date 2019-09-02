function d = indegree (H)
%INDEGREE in-degree of a directed or undirected gbgraph.
% d = indegree (H) is a vector of size n for gbgraph with n nodes equal to d =
% sum (spones (H)).  d(i) is the number of entries in H(:,i).
%
% See also digraph/indegree, gb/degree, gb/outdegree,

% TODO tests

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

if (isundirected (H))
    d = degree (H) ;
else
    d = gb.coldegree (H) ;
end

