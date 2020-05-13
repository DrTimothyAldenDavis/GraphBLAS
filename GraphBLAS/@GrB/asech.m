function C = asech (G)
%ASECH inverse hyperbolic secant.
% C = asech (G) computes the inverse hyperbolic secant of each entry of a
% GraphBLAS matrix G.  Since asech (0) is nonzero, the result is a full
% matrix.  C is complex if G is complex, or if any real entries are
% outside of the range [0,1].
%
% See also GrB/sec, GrB/asec, GrB/sech.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights
% Reserved. http://suitesparse.com.  See GraphBLAS/Doc/License.txt.

if (~isfloat (G))
    G = GrB (G, 'double') ;
end

C = acosh (GrB.apply ('minv', full (G))) ;

