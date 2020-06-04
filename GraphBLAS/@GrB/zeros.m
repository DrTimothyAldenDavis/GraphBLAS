function C = zeros (arg1, arg2, arg3, arg4)
%ZEROS an all-zero matrix, the same type as G.
% C = zeros (m, n, 'like', G) or C = zeros ([m n], 'like', G) returns
% an m-by-n matrix with no entries, of the same type as G.
%
% See also GrB/ones, GrB/false, GrB/true.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights
% Reserved. http://suitesparse.com.  See GraphBLAS/Doc/License.txt.

if (nargin == 4)

    if (~isequal (arg3, 'like'))
        gb_error ('usage: zeros (m, n, ''like'', G)') ;
    end
    C = GrB (arg1, arg2, gbtype (arg4.opaque)) ;

elseif (nargin == 3)

    if (~isequal (arg2, 'like'))
        gb_error ('usage: zeros ([m n], ''like'', G)') ;
    end
    C = GrB (arg1 (1), arg1 (2), gbtype (arg3.opaque)) ;

else

    gb_error ('usage: zeros (m, n, ''like'', G)') ;

end

