function C = sech (G)
%SECH hyperbolic secant.
% C = sech (G) computes the hyperbolic secant of each entry of a
% GraphBLAS matrix G.  Since sech(0) is nonzero, C is a full matrix.
%
% See also GrB/sec, GrB/asec, GrB/asech.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights
% Reserved. http://suitesparse.com.  See GraphBLAS/Doc/License.txt.

C = GrB.apply ('minv', cosh (G)) ;

