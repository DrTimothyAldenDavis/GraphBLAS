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
% The inputs are MATLAB matrices or GraphBLAS structs.  The output
% is a GraphBLAS struct.
%
% See also GrB/lt, GrB/min, GrB/max, GrB/ne, GrB/pow2, GrB/atan2,
% GrB/bitset.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights
% Reserved. http://suitesparse.com.  See GraphBLAS/Doc/License.txt.

% FUTURE: this would be faster if a variant to GrB_eWiseAdd would
% be added, which takes 2 scalar values (one for A and one for B),
% and applies the binary operator on the union of A and B.
%
% The following operations could use this feature:
%
%   lt, gt, min, max, ne, pow2, atan2, bitset: (0, 0)
%   bitshift: (0, (int8) 0)
%   minus: (0, 0), using the '-' operator instead of A+(-B)
%   complex: (0, 0)
%   hypot: (0, 0)
%
% Suppose the variant is GrB_eWiseUnion (... A, ascalar, B, bscalar).
% If a scalar is NULL, it would be treated as eWiseAdd, with
% f(aij,NULL) = aij (the 'first' operator), and f(NULL,bij)=bij,
% the 'second' operator.  So if both are passed in as NULL,
% GrB_eWiseUnion (... A, NULL, B, NULL, ...) is identical to
% GrB_eWiseAdd (... A, B...).

ctype = gboptype (gbtype (A), gbtype (B)) ;

% A0 = expand A by padding it with zeros from the pattern of B 
A0 = gbeadd (['1st.' ctype], A, gb_expand (0, B, ctype)) ;

% B0 = expand B by padding it with zeros from the pattern of A 
B0 = gbeadd (['1st.' ctype], B, gb_expand (0, A, ctype)) ;

% A0 and B0 now have the same pattern, so gbemult can be used:
C = gbemult (A0, op, B0) ;

