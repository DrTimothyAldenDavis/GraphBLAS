function k = kind (G)
%KIND Return the kind of a gbgraph.
% k = kind (G) returns the kind of the gbgraph G, either 'directed'
% or 'undirected.'
%
% See also graph, digraph, gbgraph.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

k = G.graphkind ;

