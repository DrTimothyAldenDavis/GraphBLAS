function C = gb_union_op (op, A, B)
%GB_SPARSE_BINOP apply a binary operator to two sparse matrices.
% The pattern of C is the set union of A and B.  A and B must first be
% expanded to include explicit zeros in the set union of A and B.  For
% example, with A < B for two matrices A and B:
%
%     in A        in B        A(i,j) < B (i,j)    true or false
%     not in A    in B        0 < B(i,j)          true or false
%     in A        not in B    A(i,j) < 0          true or false
%     not in A    not in B    0 < 0               false, not in C
%
% A and B are expanded to the set union of A and B, with explicit zeros,
% and then the op is applied via GrB.emult.  Unlike the built-in
% GraphBLAS GrB.eadd and GrB.emult, both of which apply the operator
% only to the set intersection, this function applies the operator to the
% set union of A and B.
%
% See also GrB/lt, GrB/min, GrB/max, GrB/ne, GrB/pow2, GrB/atan2.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights
% Reserved. http://suitesparse.com.  See GraphBLAS/Doc/License.txt.

ctype = GrB.optype (A, B) ;

% A0 = expand A by padding it with zeros from the pattern of B 
A0 = GrB.eadd (['1st.' ctype], A, GrB.expand (GrB (0, ctype), B)) ;

% B0 = expand B by padding it with zeros from the pattern of A 
B0 = GrB.eadd (['1st.' ctype], B, GrB.expand (GrB (0, ctype), A)) ;

% A0 and B0 now have the same pattern, so GrB.emult can be used:
C = GrB.emult (A0, op, B0) ;

