function C = acosh (G)
%ACOSH inverse hyperbolic cosine.
% C = acosh (G) is the inverse hyperbolic cosine of each entry G.  Since
% acosh (0) is nonzero, the result is a full matrix.  C is complex if
% any (G < 1).
%
% See also GhB/cos, GhB/acos, GhB/cosh.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

ghb = 1 ;     % 0 for GrB, 1 for GhB

C = gb_trig (ghb, 'acosh', gzb_full (ghb, G)) ;

