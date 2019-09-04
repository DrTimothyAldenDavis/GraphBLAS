function C = prune (G, varargin)
%PRUNE remove explicit values from a gbgraph matrix.
% C = prune (G) removes edges with weight zero from G.
% C = prune (G, s) removes edges with weight equal to id.
%
% See also gb.prune, gb/full, gbgraph/pruneself.

C = gbgraph (gb.prune (G, varargin {:}), kind (G)) ;

