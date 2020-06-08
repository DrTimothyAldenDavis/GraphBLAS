function I = gb_subsindex (G, offset)
%GB_SUBSINDEX subscript index from GraphBLAS matrix
% implements I = subsindex (G)

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights
% Reserved. http://suitesparse.com.  See GraphBLAS/Doc/License.txt.

% TODO can this handle logical indexing too?

type = gbtype (G) ;
[m, n] = gbsize (G) ;

I = gbextractvalues (G) ;

if ((isequal (type, 'double') || isequal (type, 'single')) ...
    && isequal (I, round (I)))
    I = int64 (I) ;
elseif (~contains (type, 'int'))
    gb_error ('array indices must be integers') ;
end

if (offset ~= 0)
    I = I - offset ;
end

if (m == 1)
    I = I' ;
elseif (n > 1)
    I = reshape (I, m, n) ;
end

