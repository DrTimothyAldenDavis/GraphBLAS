function C = acoth (G)
%ACOTH inverse hyperbolic cotangent.
% C = acoth (G) computes the inverse hyberbolic cotangent of each entry of
% a GraphBLAS matrix G.  C is complex if G is complex, or if any
% abs(G)<1.  Since acoth (0) is nonozero, C is a full matrix.
%
% See also GrB/cot, GrB/acot, GrB/coth.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights
% Reserved. http://suitesparse.com.  See GraphBLAS/Doc/License.txt.

if (~isfloat (G))
    G = GrB (G, 'double') ;
end

C = atanh (GrB.apply ('minv', full (G))) ;

