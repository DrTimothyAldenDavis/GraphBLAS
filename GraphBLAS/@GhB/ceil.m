function C = ceil (G)
%CEIL round entries of a matrix to nearest integers towards infinity.
%
% See also GhB/floor, GhB/round, GhB/fix.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

ghb = 1 ;     % 0 for GrB, 1 for GhB

C = gb_ceil (ghb, G) ;

