function C = digraph (H)
%DIGRAPH convert a gbgraph into a MATLAB digraph.
% C = digraph (H) converts a gbgraph H into a directed MATLAB digraph.  H may
% be directed or undirected.  No weights are added if H is logical.

% TODO test

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

type = gb.type (H) ;

if (isequal (type, 'logical'))
    C = digraph (logical (H)) ;
elseif (isequal (type, 'double'))
    C = digraph (double (H)) ;
else
    % all other types (int*, uint*, single)
    [i, j, x] = gb.extracttuples (H) ;
    C = digraph (i, j, x) ;
end

