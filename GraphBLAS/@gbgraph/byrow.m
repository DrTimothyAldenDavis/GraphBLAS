function G = byrow (G)
%BYROW Change the format of the gbgraph G to 'by row'
% G = byrow (G) converts G to the format 'by row'.
% The graph kind ('undirected' or 'directed') is not changed.
%
% See also gbgraph/bycol, gb.format.

if (gb.isbycol (G))
    G = gbgraph (G, 'by row', kind (G), false) ;
end

