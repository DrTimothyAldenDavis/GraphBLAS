function C = le (A, B)
%A <= B less than or equal to.
% C = (A <= B) compares A and B element-by-element.  One or
% both may be scalars.  Otherwise, A and B must have the same size.
%
% See also GhB/lt, GhB/gt, GhB/ge, GhB/ne, GhB/eq.

% The pattern of C depends on the type of inputs:
% A scalar, B scalar:  C is scalar.
% A scalar, B matrix:  C is full if A<=0, otherwise C is a subset of B.
% B scalar, A matrix:  C is full if B>=0, otherwise C is a subset of A.
% A matrix, B matrix:  C is full.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

C = gb_le (1, A, B) ;

