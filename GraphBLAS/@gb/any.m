function C = any (G, option)
%ANY True if any element of a GraphBLAS matrix is nonzero or true.
%
% C = any (G) is true if any entry in G is nonzero or true.  If G is a
% matrix, C is a row vector with C(j) = any (G (:,j)).
%
% C = any (G, 'all') is a scalar, true if any entry in G is nonzero or true.
% C = any (G, 1) is a row vector with C(j) = any (G (:,j))
% C = any (G, 2) is a column vector with C(i) = any (G (i,:))
%
% See also all, nnz, gb.nvals.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

[m n] = size (G) ;

if (nargin == 1)

    % C = any (G)
    if (isvector (G))
        % C = any (G) for a vector G results in a scalar C
        C = gb.reduce ('|.logical', G) ;
    else
        % C = any (G) reduces each column to a scalar,
        % giving a 1-by-n row vector.
        C = gb.vreduce ('|.logical', G, struct ('in0', 'transpose'))' ;
    end

elseif (nargin == 2)

    % C = any (G, option)
    if (isequal (option, 'all'))
        % C = any (G, 'all'), reducing all entries to a scalar
        C = gb.reduce ('|.logical', G) ;
    elseif (isequal (option, 1))
        % C = any (G, 1) reduces each column to a scalar,
        % giving a 1-by-n row vector.
        C = gb.vreduce ('|.logical', G, struct ('in0', 'transpose'))' ;
    elseif (isequal (option, 2))
        % C = any (G, 2) reduces each row to a scalar,
        % giving an m-by-1 column vector.
        C = gb.vreduce ('|.logical', G) ;
    else
        error ('unknown option') ;
    end

else
    error ('invalid usage') ;
end

