function x = gb_scalar (A)
%GB_SCALAR get contents of a scalar
% x = gb_scalar (A).  A may be a MATLAB scalar or a GraphBLAS
% scalar as a struct (not an object).

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights
% Reserved. http://suitesparse.com.  See GraphBLAS/Doc/License.txt.

assert (~isobject (A)) ;
assert (isequal (gbsize (A), [1 1]))

[~, ~, x] = gbextracttuples (A) ;
if (isempty (x))
    x = 0 ;
else
    x = x (1) ;
end

