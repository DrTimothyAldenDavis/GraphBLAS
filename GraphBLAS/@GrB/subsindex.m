function I = subsindex (G)
%SUBSINDEX subscript index from GraphBLAS matrix
% I = subsindex (G) is used when the GraphBLAS matrix G is used to
% index into a non-GraphBLAS matrix A, for A(G).
%
% See also GrB/subsref, GrB/subsasgn.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights
% Reserved. http://suitesparse.com.  See GraphBLAS/Doc/License.txt.

G = G.opaque ;
type = gbtype (G) ;
[m, n] = gbsize (G) ;

I = gbextractvalues (G) ;

if (contains (type, 'int'))
    I = I - 1 ;
elseif ((isequal (type, 'double') || isequal (type, 'single')) ...
    && isequal (I, round (I)))
    I = int64 (I) - 1 ;
else
    error ('array indices must be integers') ;
end

if (m == 1)
    I = I' ;
elseif (n > 1)
    I = reshape (I, m, n) ;
end

