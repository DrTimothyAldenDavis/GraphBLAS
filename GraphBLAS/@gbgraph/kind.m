function k = kind (H)
%KIND Return the kind of a gbgraph.
% k = gbgraph.kind (H) returns the kind of the gbgraph H, either 'directed'
% or 'undirected.'
%
% See also graph, digraph, gbgraph.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

k = H.graphkind ;

