function C = acos (G)
%ACOS inverse cosine.
% C = acos (G) computes the inverse cosine of each entry of a GraphBLAS
% matrix G.  Since acos (0) is nonzero, the result is a full matrix.
% C is complex if any (abs(G) > 1).
%
% See also GrB/cos, GrB/cosh, GrB/acosh.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights
% Reserved. http://suitesparse.com.  See GraphBLAS/Doc/License.txt.

C = GrB (gb_trig ('acos', G.opaque)) ;

