function C = erf (G)
%ERF error function.
% C = erf (G) computes the error function of each entry of a GraphBLAS
% matrix G.  G must be real.
%
% See also GrB/erfc, erfcx, erfinv, erfcinv.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights
% Reserved. http://suitesparse.com.  See GraphBLAS/Doc/License.txt.

if (~isreal (G))
    error ('input must be real') ;
end

if (~isfloat (G))
    G = GrB (G, 'double') ;
end

C = GrB.apply ('erf', G) ;

