function C = cos (G)
%COS cosine.
% C = cos (G) computes the cosine of each entry of a GraphBLAS matrix G.
% Since cos (0) = 1, the result is a full matrix.
%
% See also GrB/acos, GrB/cosh, GrB/acosh.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights
% Reserved. http://suitesparse.com.  See GraphBLAS/Doc/License.txt.

if (~isfloat (G))
    G = GrB (G, 'double') ;
end

C = GrB.apply ('cos', full (G)) ;

