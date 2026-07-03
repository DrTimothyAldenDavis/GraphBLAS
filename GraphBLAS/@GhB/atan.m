function C = atan (G)
%ATAN inverse tangent.
% C = atan (G) is the inverse tangent of each entry of G.
%
% See also GhB/tan, GhB/tanh, GhB/atanh, GhB/atan2.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

ghb = 1 ;     % 0 for GrB, 1 for GhB

C = gb_atan (ghb, G) ;

