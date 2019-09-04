function G = underlying (G)
%UNDERLYING constructs the underlying undirected graph.
% G = underlying (G) converts G into its underlying undirected graph.
% If G is already undirected, then it is returned unmodified.
%
% See also gbgraph/pruneself.m

if (isdirected (G))
    G = gbgraph (G+G', 'undirected') ;
end

