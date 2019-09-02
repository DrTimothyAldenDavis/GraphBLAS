function H = bycol (H)
%BYCOL Change the format of the gbgraph H to 'by col'
%
% See also gbgraph/byrow, gb.format.

if (gb.isbyrow (H))
    H = gbgraph (gb (H, 'by col'), H.graphkind, false) ;
end

