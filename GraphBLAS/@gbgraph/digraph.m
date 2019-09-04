function C = digraph (G)
%DIGRAPH convert a gbgraph into a MATLAB digraph.
% C = digraph (G) converts a gbgraph G into a directed MATLAB digraph C.  G may
% be directed or undirected.  No weights are added if G is logical.

% TODO test

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

type = gb.type (G) ;

if (isequal (type, 'logical'))
    C = digraph (logical (G)) ;
elseif (isequal (type, 'double'))
    C = digraph (double (G)) ;
else
    % all other types (int*, uint*, single)
    n = numnodes (G) ;
    [i, j, x] = gb.extracttuples (G) ;
    C = digraph (i, j, x, n) ;
end

