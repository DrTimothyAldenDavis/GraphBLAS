function C = sqrt (G)
%SQRT Square root.
% C = sqrt (G) is the square root of the elements of the GraphBLAS matrix
% G.
%
% See also GrB.apply, GrB/hypot.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights
% Reserved. http://suitesparse.com.  See GraphBLAS/Doc/License.txt.

C = GrB (gb_to_real_if_imag_zero (gb_trig ('sqrt', G.opaque))) ;

