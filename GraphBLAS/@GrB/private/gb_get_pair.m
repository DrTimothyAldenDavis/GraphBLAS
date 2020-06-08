function [x, y] = gb_get_pair (A)
%GB_GET_PAIR get a pair of scalars from a parameter of length 2

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights
% Reserved. http://suitesparse.com.  See GraphBLAS/Doc/License.txt.

if (isobject (A))
    A = A.opaque ;
end

[m, n] = gbsize (A) ;
if (m*n ~= 2)
    gb_error ('input parameter %s must have length 2', inputname (1)) ;
end

A = gbfull (A) ;
x = A (1) ;
y = A (2) ;

