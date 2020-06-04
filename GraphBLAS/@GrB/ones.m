function C = ones (arg1, arg2, arg3, arg4)
%ONES a matrix with all ones, the same type as G.
% C = ones (m, n, 'like', G) or C = ones ([m n], 'like', G) constructs an
% m-by-n GraphBLAS matrix C with all entries equal to one.  C has the
% same type as G.
%
% See also GrB/zeros, GrB/false, GrB/true.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights
% Reserved. http://suitesparse.com.  See GraphBLAS/Doc/License.txt.

if (nargin == 4)

    if (~isequal (arg3, 'like'))
        gb_error ('usage: ones (m, n, ''like'', G)') ;
    end
    Z = gbnew (arg1, arg2, gbtype (arg4.opaque)) ;

elseif (nargin == 3)

    if (~isequal (arg2, 'like'))
        gb_error ('usage: ones ([m n], ''like'', G)') ;
    end
    Z = gbnew (arg1 (1), arg1 (2), gbtype (arg3.opaque)) ;

else

    gb_error ('usage: ones (m, n, ''like'', G)') ;

end

C = GrB (gbsubassign (Z, 1)) ;

