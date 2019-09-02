function [e d] = numedges (H)
%NUMEDGES number of edges in a GraphBLAS gbgraph.
% e = numedges (H) is the number of edges in a GraphBLAS gbgraph.  If H is a
% directed graph, then this is the same as nnz (H).  Otherise, if H is an
% undirected graph, each pair of off-diagonal entries in the adjacency matrix
% is only counted once.
%
% [e d] = numedges (H) also returns the number of self-edges, d.
%
% See also graph/numedges, digraph/numedges.

% TODO: tests

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

e = nnz (H) ;

if (isundirected (H))
    d = nnz (diag (H)) ;
    e = (e - d) / 2 + d ;
elseif (nargout > 1)
    d = nnz (diag (H)) ;
end

