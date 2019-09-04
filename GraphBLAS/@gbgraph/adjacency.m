function G = adjacency (G, arg)
%ADJACENCY Graph adjacency matrix.
% G = adjacency (G) returns an n-by-n logical GraphBLAS matrix G, which is the
% adjacency matrix of G (with n nodes), where G(i,j) = 1 if (i,j) is an edge in
% the gbgraph G.
% 
% G = adjacency (G, 'weighted') returns G as a weighted graph, of the same
% type as G.
%
% See also graph/adjacency, digraph/adjacency.

% TODO tests

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

if (nargin < 2)
    G = gb (G, 'logical') ;
elseif (isequal (arg, 'weighted'))
    G = gb (G) ;
else
    error ('G = adjacency (G,W) not yet supported') ;
end

