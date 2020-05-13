function C = sec (G)
%SEC secant.
% C = sec (G) computes the secant of each entry of a GraphBLAS matrix G.
% Since sec (0) = 1, the result is a full matrix.
%
% See also GrB/asec, GrB/sech, GrB/asech.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights
% Reserved. http://suitesparse.com.  See GraphBLAS/Doc/License.txt.

C = GrB.apply ('minv', cos (G)) ;

