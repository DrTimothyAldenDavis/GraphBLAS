function C = csch (G)
%CSCH hyperbolic cosecant.
% C = csch (G) computes the hyperbolic cosecant of each entry of a
% GraphBLAS matrix G.  Since csch(0) is nonzero, C is a full matrix.
%
% See also GrB/csc, GrB/acsc, GrB/acsch.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights
% Reserved. http://suitesparse.com.  See GraphBLAS/Doc/License.txt.

C = GrB.apply ('minv', full (sinh (G))) ;

