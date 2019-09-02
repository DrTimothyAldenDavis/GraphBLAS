function C = reordernodes (H, p)
%REORDERNODES Reorder nodes of a gbgraph.
% C = reordernodes (H, p) returns a graph C with the nodes reordered.  The
% adjacency matrix of the new graph is H (p, p).  C is not defined if p is not
% a permutation of 1:n, where n = numnodes (H); this condition is not checked.
%
% See also graph/reordernodes, digraph/reordernodes.

% TODO: tests

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

if (~iscell (p))
    p = { p } ;
end

C = gbgraph (gb.extract (H, p, p), H.graphkind, false) ;

