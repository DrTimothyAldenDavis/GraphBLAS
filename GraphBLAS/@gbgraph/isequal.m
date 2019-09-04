function s = isequal (G1, G2)
%ISEQUAL determine if two gbgraphs G1 and G2 are equal.
% s = isequal (G1, G2) returns true if the gbraphs G1 and G2 are equal.  To be
% equal, the graphs G1 and G2 must both be gbgraphs and have the same kind (both
% directed or both undirected), and also must have equal adjacency matrices.
%
% See also gb/isequal.

if (~isequal (class (G1), class (G2)))
    s = false ;
elseif (~isequal (kind (G1), kind (G2)))
    s = false ;
else
    s = isequal@gb (G1, G2) ;
end

