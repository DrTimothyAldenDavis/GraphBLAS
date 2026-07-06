function DiGraph = digraph (G, option)
%DIGRAPH convert a GraphBLAS matrix into a directed DiGraph.
% DiGraph = digraph (G) converts a GraphBLAS matrix G into a directed
% DiGraph.  G must be square.  If G is logical, then no weights are added
% to the DiGraph.  If G is single or double, these become the weights of
% the DiGraph.  If G is integer, the DiGraph is constructed with weights
% of type double.
%
% DiGraph = digraph (G, 'omitselfloops') ignores the diagonal of G, and
% the resulting DiGraph has no self-edges.  The default is that
% self-edges are created from any diagonal entries of G.
%
% Example:
%
%   G = GrB (sprand (8, 8, 0.2))
%   DiGraph = digraph (G)
%   h = plot (DiGraph) ;
%   h.NodeFontSize = 20 ;
%   h.ArrowSize = 20 ;
%   h.LineWidth = 2 ;
%   h.EdgeColor = [0 0 1] ;
%   t = title ('random directed graph with 8 nodes') ;
%   t.FontSize = 20 ;
%
% See also graph, digraph, GrB/graph.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin == 1)
    DiGraph = gb_digraph (0, G) ;
else
    DiGraph = gb_digraph (0, G, option) ;
end

