function c = edgecount (H, s, t)
%EDGECOUNT Determine the number of edges between two nodes of a gbgraph.
% c = edgecount (H, s, t) determines the number of edges between two nodes s
% and t.  If s and t are scalars, c is 0 or 1 (multigraphs are not supported).
% If s and t are lists of indices, then c is the number of edges in the
% H(s,t) submatrix of the adjacency matrix H.
%
% See also graph/edgecount, digraph/edgecount.

% TODO: tests

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

if (~iscell (s))
    s = { s } ;
end
if (~iscell (t))
    t = { t } ;
end

c = nnz (gb.extract (H, s, t)) ;

