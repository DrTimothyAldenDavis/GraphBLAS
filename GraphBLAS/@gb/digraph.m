function C = digraph (G)
% C = digraph (G) is the directed MATLAB graph of the GraphBLAS matrix G.
% G is used as the adjacency matrix of the digraph C.  G must be square.
% No weights are added if G is logical.

% TODO test

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

[m n] = size (G) ;
if (m ~= n)
    error ('G must be square') ;
end

type = gb.type (G) ;

if (isequal (type, 'logical'))
    C = digraph (logical (G)) ;
elseif (isequal (type, 'double'))
    C = digraph (double (G)) ;
else
    % all other types (int*, uint*, single)
    [i, j, x] = gb.extracttuples (G) ;
    C = digraph (i, j, x) ;
end

