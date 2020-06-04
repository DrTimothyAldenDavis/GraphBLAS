function C = acosh (G)
%ACOSH inverse hyperbolic cosine.
% C = acosh (G) computes the inverse hyperbolic cosine of each entry of a
% GraphBLAS matrix G.  Since acosh (0) is nonzero, the result is a full
% matrix.  C is complex if any (G < 1).
%
% See also GrB/cos, GrB/acos, GrB/cosh.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights
% Reserved. http://suitesparse.com.  See GraphBLAS/Doc/License.txt.

C = GrB (gb_trig ('acosh', gbfull (G.opaque))) ;

