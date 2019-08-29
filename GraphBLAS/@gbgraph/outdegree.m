function d = outdegree (G)
%OUTDEGREE out-degree of a GraphBLAS matrix
% d = outdegree (G) is a vector of size m for an m-by-n GraphBLAS matrix G,
% equal to d = sum (spones (G')).
%
% See also digraph/outdegree, gb/degree, gb/indegree.

% TODO tests

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

d = row_degree (G) ;

