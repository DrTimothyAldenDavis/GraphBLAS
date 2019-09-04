function C = reordernodes (G, p)
%REORDERNODES Reorder nodes of a gbgraph.
% C = reordernodes (G, p) returns a graph C with the nodes reordered.  The
% adjacency matrix of the new graph is G (p, p).  C is not defined if p is not
% a permutation of 1:n, where n = numnodes (G); this condition is not checked.
%
% See also graph/reordernodes, digraph/reordernodes.

% TODO: tests

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

if (~iscell (p))
    p = { p } ;
end

C = gbgraph (gb.extract (G, p, p), kind (G), false) ;

