function e = nself (G)
%NSELF number of self-edges in a GraphBLAS gbgraph.
% e = nself (G) is the number of edges in a GraphBLAS gbgraph.
%
% See also gbgraph/numedges, gbgraph/pruneself, graph/numedges.

% TODO: tests

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

e = nnz (diag (G)) ;

