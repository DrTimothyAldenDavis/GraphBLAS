function C = acsc (G)
%ACSC inverse cosecant.
% C = acsc (G) computes the inverse cosecant of each entry of a GraphBLAS
% matrix G.  Since acsc (0) is nonzero, C is a full matrix.
%
% See also GrB/csc, GrB/csch, GrB/acsch.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights
% Reserved. http://suitesparse.com.  See GraphBLAS/Doc/License.txt.

if (~isfloat (G))
    G = GrB (G, 'double') ;
end

C = asin (GrB.apply ('minv', full (G))) ;

