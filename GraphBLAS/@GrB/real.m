function C = real (G)
%REAL complex real part.
% C = real (G) returns the real part of the GraphBLAS matrix G.
%
% See also GrB/conj, GrB/imag.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights
% Reserved. http://suitesparse.com.  See GraphBLAS/Doc/License.txt.

if (isreal (G))
    C = G ;
else
    C = GrB.apply ('creal', G) ;
end

