function Graph = graph (G, varargin)
%GRAPH convert a GraphBLAS matrix into a undirected Graph.
% Graph = graph (G) converts a GraphBLAS matrix G into an undirected
% Graph.  G is assumed to be symmetric; only tril (G) is used by default.
% G must be square.  If G is logical, then no weights are added to the
% Graph.  If G is single or double, these become the weights of the
% Graph.  If G is integer, the Graph is constructed with weights of type
% double.
%
% Graph = graph (G, ..., 'upper') uses triu (G) to construct the Graph.
% Graph = graph (G, ..., 'lower') uses tril (G) to construct the Graph.
% The default is 'lower'.
%
% Graph = graph (G, ..., 'omitselfloops') ignores the diagonal of G, and
% the resulting Graph has no self-edges.  The default is that
% self-edges are created from any diagonal entries of G.
%
% Example:
%
%   G = GhB (bucky) ;
%   Graph = graph (G)
%   plot (Graph)
%
% See also graph, digraph, GhB/digraph.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

Graph = gb_graph (1, G, varargin {:}) ;

