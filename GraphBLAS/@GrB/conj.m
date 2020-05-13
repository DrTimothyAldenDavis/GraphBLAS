function C = conj (G)
%CONJ complex conjugate of a GraphBLAS matrix.
%
% See also GrB/real, GrB/imag.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights
% Reserved. http://suitesparse.com.  See GraphBLAS/Doc/License.txt.

if (isreal (G))
    C = G ;
else
    C = GrB.apply ('conj', G) ;
end

