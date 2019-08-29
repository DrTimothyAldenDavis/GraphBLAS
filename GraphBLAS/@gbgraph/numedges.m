function [e d] = numedges (H)
%NUMEDGES number of edges in a GraphBLAS graph.
%
% numedges(H) is the number of edges in a GraphBLAS graph.  If it is digraph, then
% this is the same as nnz(H).  Otherise, if it is an undirected graph, pairs of
% off-diagonal entries in the adjacency matrix are only counted once.
%
% [e d] = numedges (H) also returns the number of self-edges, d.

e = nnz (H) ;
d = nnz (diag (H)) ;
if (isequal (H.graphkind, 'graph'))
    e = (e - d) / 2 + d ;
end

