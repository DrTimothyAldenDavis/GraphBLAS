function C = log (G)
%LOG natural logarithm of the entries of a GraphBLAS matrix.
% C = log (G) computes the natural logarithm of each entry of a GraphBLAS
% matrix G.  Since log (0) is nonzero, the result is a full matrix.
% If any entry in G is negative, the result is complex.
%
% See also GrB/log1p, GrB/log2, GrB/log10, GrB/exp.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights
% Reserved. http://suitesparse.com.  See GraphBLAS/Doc/License.txt.

G = G.opaque ;
C = GrB (gb_to_real_if_imag_zero (gb_trig ('log', gbfull (G)))) ;

