function n = nmalloc
%NMALLOC number of malloc's in GraphBLAS, for testing/development only

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

ghb = 1 ;     % 0 for GrB, 1 for GhB

n = gbmex_nmalloc ;
