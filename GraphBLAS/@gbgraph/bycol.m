function G = bycol (G)
%BYCOL Change the format of the gbgraph G to 'by col'
% G = bycol (G) converts G to the format 'by col'.
% The graph kind ('undirected' or 'directed') is not changed.
%
% See also gbgraph/byrow, gb.format.

if (gb.isbyrow (G))
    G = gbgraph (G, 'by col', kind (G), false) ;
end

