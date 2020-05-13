function C = cosh (G)
%COSH hyperbolic cosine.
% C = cosh (G) computes the hyperbolic cosine of each entry of a GraphBLAS
% matrix G.  Since cosh (0) = 1, the result is a full matrix.
%
% See also GrB/cos, GrB/acos, GrB/acosh.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights
% Reserved. http://suitesparse.com.  See GraphBLAS/Doc/License.txt.

if (~isfloat (G))
    G = GrB (G, 'double') ;
end

C = GrB.apply ('cosh', full (G)) ;

