function C = reordernodes (H, order)
%REORDERNODES Reorder nodes of a gbgraph.
% C = reordernodes (H, order) returns a graph C with the nodes reordered.  The
% adjacency matrix of the new graph is H (order, order).  C is not defined if
% order is not a permutation of 1:n, where n = numnodes (H).
%
% See also graph/reordernodes, digraph/reordernodes.

% TODO: tests

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

if (~iscell (order))
    order = { order } ;
end

C = gbgraph (gb.extract (H, order, order), H.graphkind, false) ;

