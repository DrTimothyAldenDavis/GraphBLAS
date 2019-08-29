function d = indegree (G)
%INDEGREE in-degree of a GraphBLAS matrix
% d = indegree (G) is a vector of size n for an m-by-n GraphBLAS matrix G,
% equal to d = sum (spones (G)).
%
% See also gb/degree, gb/outdegree, graph/indegree.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

d = col_degree (G) ;

