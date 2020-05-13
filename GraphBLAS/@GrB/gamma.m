function C = gamma (G)
%GAMMA gamma function.
% C = gamma (G) computes the gamma of each entry of a GraphBLAS matrix G.
% Since gamma (0) = inf, the result is a full matrix.  G must be real.
%
% See also GrB/gammaln.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights
% Reserved. http://suitesparse.com.  See GraphBLAS/Doc/License.txt.

if (~isreal (G))
    error ('input must be real') ;
end

if (~isfloat (G))
    G = GrB (G, 'double') ;
end

C = GrB.apply ('gamma', full (G)) ;

