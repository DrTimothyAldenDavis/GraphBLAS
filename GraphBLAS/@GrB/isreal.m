function s = isreal (G)
%ISREAL true for real matrices.
% isreal (G) is true for a GraphBLAS matrix G, unless it has a type of
% 'single complex' or 'double complex'.
%
% See also GrB/isnumeric, GrB/isfloat, GrB/isinteger, GrB/islogical,
% GrB.type, GrB/isa, GrB.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

ghb = 1 ;     % 0 for GrB, 1 for GhB

s = ~gb_contains (gbmex_type (G), 'complex') ;

