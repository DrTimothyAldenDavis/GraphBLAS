function C = or (A, B)
%| logical OR.
% C = (A | B) is the element-by-element logical OR of A and B.  One or
% both may be scalars.  Otherwise, A and B must have the same size.
%
% See also GrB/and, GrB/xor, GrB/not.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

ghb = 1 ;     % 0 for GrB, 1 for GhB

C = gb_or (ghb, A, B) ;

