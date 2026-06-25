function assert (G)
%ASSERT generate an error when a condition is violated.
%
% See also error.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

ghb = 0 ;     % 0 for GrB, 1 for GhB

builtin ('assert', logical (G)) ;

