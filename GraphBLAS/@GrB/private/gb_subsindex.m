function I = gb_subsindex (G, offset)
%GB_SUBSINDEX subscript index from GraphBLAS matrix
% Implements I = subsindex (G) for X(G) when X is a MATLAB matrix.
% Explicit zeros must be kept.  I must contain entries in range 0 to
% prod (size (X)) - 1.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights
% Reserved. http://suitesparse.com.  See GraphBLAS/Doc/License.txt.

[m, n, type] = gbsize (G) ;

if (isequal (type, 'double') || isequal (type, 'single'))

    I = gbextractvalues (G) ;
    if (~isequal (I, round (I)))
        error ('array indices must be integers') ;
    end
    I = int64 (I) ;

elseif (contains (type, 'int'))

    % Explicit zeros must be kept.  I must contain entries in range 0 to
    % prod (size (X)) - 1.  The type of I is also kept.
    I = gbextractvalues (G) ;

else

    error ('array indices must be integers') ;

end


% for zero-based indexing, offset is zero.
if (offset ~= 0)
    % otherwise, subtract the offset.
    I = I - offset ;
end

% reshape I as needed
if (m == 1)
    I = I' ;
elseif (n > 1)
    I = reshape (I, m, n) ;
end

