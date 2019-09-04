function [handle titlehandle] = plot (G, varargin)
%PLOT Draw a GraphBLAS gbgraph.
% plot (G, ...) plots the GraphBLAS gbgraph G.  See 'help graph/plot' or 'help
% digraph/plot' for a list of line, marker, and axis options.
%
% h = plot (...) returns the handle to the plot.  [h t] = plot (...) also
% returns the handle to the title of the plot.  Use get(h) and get(t) to show
% all properties that can be adjusted for these two handles.
%
% Example:
%
%   G = gbgraph (bucky) ;
%   [h t] = plot (G) ;
%   h.NodeFontSize = 20 ;
%   t.FontSize = 20 ;
%   h.LineWidth = 4 ;
%
% See also graph/plot, digraph/plot.

% TODO test

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

[e d] = numedges (G) ;
n = numnodes (G) ;
k = kind (G) ;

if (isequal (k, 'undirected'))
    h = plot (graph (G), varargin {:}) ;
else
    h = plot (digraph (G), varargin {:}) ;
end

t = title (sprintf (...
    'GraphBLAS: %s graph, %d nodes %d edges (%d self-edges)\n', ...
    k, n, e, d)) ;

if (nargout > 0)
    handle = h ;
end

if (nargout > 1)
    titlehandle = t ;
end

