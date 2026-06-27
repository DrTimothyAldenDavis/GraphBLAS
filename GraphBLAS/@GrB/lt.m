function C = lt (A, B)
%A < B less than.
% C = (A < B) compares A and B element-by-element.  One or
% both may be scalars.  Otherwise, A and B must have the same size.
%
% See also GrB/le, GrB/gt, GrB/ge, GrB/ne, GrB/eq.

% The pattern of C depends on the type of inputs:
% A scalar, B scalar:  C is scalar.
% A scalar, B matrix:  C is full if A<0, otherwise C is a subset of B.
% B scalar, A matrix:  C is full if B>0, otherwise C is a subset of A.
% A matrix, B matrix:  C has the pattern of the set union, A+B.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

ghb = 0 ;     % 0 for GrB, 1 for GhB

C = gb_lt (ghb, A, B) ;

