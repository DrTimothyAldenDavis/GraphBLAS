function C = empty (arg1, arg2)
%GB.EMPTY construct an empty GraphBLAS sparse matrix
% C = gb.empty is a 0-by-0 empty matrix.
% C = gb.empty (m) is an m-by-0 empty matrix.
% C = gb.empty ([m n]) or gb.empty (m,n) is an m-by-n empty matrix, where
% one of m or n must be zero.
%
% All matrices are constructed with the 'double' type.  Use gb (m,n,type)
% to construct empty single, int*, uint*, and logical m-by-n matrices.
%
% See also gb.

m = 0 ;
n = 0 ;
if (nargin == 1)
    if (length (arg1) == 1)
        m = arg1 (1) ;
    elseif (length (arg1) == 2)
        m = arg1 (1) ;
        n = arg1 (2) ;
    else
        error ('invalid dimensions') ;
    end
elseif (nargin == 2)
    m = arg1 ;
    n = arg2 ;
end
if (~ ((m == 0) || (n == 0)))
    error ('At least one dimension must be zero') ;
end

C = gb (m, n) ;

