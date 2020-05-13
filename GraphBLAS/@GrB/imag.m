function C = imag (G)
%IMAG complex imaginary part.
% C = imag (G) returns the imaginary part of the GraphBLAS matrix G.
%
% See also GrB/conj, GrB/real.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights
% Reserved. http://suitesparse.com.  See GraphBLAS/Doc/License.txt.

if (isreal (G))
    [m, n] = size (G) ;
    C = GrB (m, n, GrB.type (G)) ;
else
    C = GrB.apply ('cimag', G) ;
end

