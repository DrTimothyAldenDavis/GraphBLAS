function [e d] = numedges (G)
%NUMEDGES number of edges in a GraphBLAS gbgraph.
% e = numedges (G) is the number of edges in a GraphBLAS gbgraph.  If G is a
% directed graph, then this is the same as nnz (G).  Otherise, if G is an
% undirected graph, each pair of off-diagonal entries in the adjacency matrix
% is only counted once.
%
% [e d] = numedges (G) also returns the number of self-edges, d.
%
% See also graph/numedges, digraph/numedges.

% TODO: tests

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

e = nnz (G) ;

if (isundirected (G))
    d = nself (G) ;
    e = (e - d) / 2 + d ;
elseif (nargout > 1)
    d = nself (G) ;
end

