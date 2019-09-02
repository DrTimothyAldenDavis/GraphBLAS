function H = byrow (H)
%BYROW Change the format of the gbgraph H to 'by row'
%
% See also gbgraph/bycol, gb.format.

if (gb.isbycol (H))
    H = gbgraph (gb (H, 'by row'), H.graphkind, false) ;
end

