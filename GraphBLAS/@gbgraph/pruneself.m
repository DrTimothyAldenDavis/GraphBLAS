function G = pruneself (G)
%PRUNESELFEDGES removes self-edges from the gbgraph G.
% G = pruneself (G) removes any self-edges from the gbgraph G.

d = nself (G) ;
if (d > 0)
    G = gbgraph (gb.select ('offdiag', G), kind (G)) ;
end

