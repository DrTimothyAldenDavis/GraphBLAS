function H = subgraph (G, I)
%SUBGRAPH Extract a subgraph from a GraphBLAS matrix.
% H = subgraph (G, I) constructs a subgraph of an n-by-n GraphBLAS matrix G,
% induced by the nodes given by I, which can either be a vector of indices in
% the range 1 to n, or a logical vector.  The latter case is the same as
% subgraph (G, find(I)).  H is returned as a GraphBLAS matrix; use graph(H) or
% digraph(H) to convert it into a MATLAB graph.

% TODO test

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

[m n] = size (G) ;
if (m ~= n)
    error ('G must be square') ;
end

if (islogical (I))
    I = find (I) ;
end

I = { I } ;
H = gb.extract (G, I, I) ;

