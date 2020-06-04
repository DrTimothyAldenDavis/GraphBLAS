function C = gammaln (G)
%GAMMALN logarithm of gamma function.
% C = gammaln (G) computes the natural logarithm of each entry of a
% GraphBLAS matrix G.  Since gammaln (0) = inf, the result is a full
% matrix.  G must be real.
%
% See also GrB/gammaln.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights
% Reserved. http://suitesparse.com.  See GraphBLAS/Doc/License.txt.

G = G.opaque ;
type = gbtype (G) ;

if (contains (type, 'complex'))
    error ('input must be real') ;
end

if (~gb_isfloat (type))
    type = 'double' ;
end

C = GrB (gbapply ('gammaln', gbfull (G, type))) ;

