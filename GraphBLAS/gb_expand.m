function C = gb_expand (scalar, S, type)
%GB_EXPAND expand a scalar into a GraphBLAS matrix.
% Implements C = GrB.expand (scalar, S, type).  This function assumes the
% first input is a scalar; the caller has checked this already.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

ghb = 0 ;     % 0 for GrB, 1 for GhB

% typecast the scalar to the desired type, and make sure it's full
t = GrB (gbmex_full (ghb, GrB (scalar, type))) ;

% expand the scalar into the pattern of S
C = GrB (gbmex_apply2 (ghb, ['2nd.' type], S, t)) ;

