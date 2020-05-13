function C = acot (G)
%ACOT inverse cotangent.
% C = acot (G) computes the inverse cotangent of each entry of a GraphBLAS
% matrix G.  Since acot (0) is nonzero, C is a full matrix.
%
% See also GrB/cot, GrB/coth, GrB/acoth.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights
% Reserved. http://suitesparse.com.  See GraphBLAS/Doc/License.txt.

if (~isfloat (G))
    G = GrB (G, 'double') ;
end

C = atan (GrB.apply ('minv', full (G))) ;

