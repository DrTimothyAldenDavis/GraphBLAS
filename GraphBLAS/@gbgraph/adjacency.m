function G = adjacency (H, arg)
%ADJACENCY Graph adjacency matrix.
% G = adjacency (H) returns an n-by-n logical GraphBLAS matrix G, which is the
% adjacency matrix of H (with n nodes), where G(i,j) = 1 if (i,j) is an edge in
% the gbgraph H.
% 
% G = adjacency (H, 'weighted') returns G as a weighted graph, of the same
% type as H.
%
% See also graph/adjacency, digraph/adjacency.

% TODO tests

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

if (nargin < 2)
    G = gb (H, 'logical') ;
elseif (isequal (arg, 'weighted'))
    G = gb (H) ;
else
    error ('G = adjacency (H,W) not yet supported') ;
end

