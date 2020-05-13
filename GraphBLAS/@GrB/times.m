function C = times (A, B)
%TIMES C = A.*B, sparse matrix element-wise multiplication.
% C = A.*B computes the element-wise multiplication of A and B.  If both
% A and B are matrices, the pattern of C is the intersection of A and B.
% If one is a scalar, the pattern of C is the same as the pattern of the
% one matrix.
%
% The input matrices may be either GraphBLAS and/or MATLAB matrices, in
% any combination.  C is returned as a GraphBLAS matrix.
%
% See also GrB/mtimes, GrB.emult, GrB.mxm.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights
% Reserved. http://suitesparse.com.  See GraphBLAS/Doc/License.txt.

C = gb_emult (A, '*', B) ;

