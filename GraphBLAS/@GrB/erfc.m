function C = erfc (G)
%ERFC complementary error function.
% C = erfc (G) computes the complementary error function of each entry of
% a GraphBLAS matrix G.  C erfc (0) = 1, the result is a full matrix.
% G must be real.
%
% See also GrB/erf, erfcx, erfinv, erfcinv.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights
% Reserved. http://suitesparse.com.  See GraphBLAS/Doc/License.txt.

if (~isreal (G))
    error ('input must be real') ;
end

if (~isfloat (G))
    G = GrB (G, 'double') ;
end

C = GrB.apply ('erfc', full (G)) ;

