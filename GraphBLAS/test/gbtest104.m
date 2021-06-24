function gbtest104
%GBTEST104 test formats

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2021, All Rights Reserved.
% SPDX-License-Identifier: GPL-3.0-or-later

rng ('default') ;
A = GrB (rand (4), 'sparse') ;
A = GrB (A, 'hypersparse')
A = GrB (A, 'bitmap')
A = GrB (A, 'full')

fprintf ('\ngbtest104: all tests passed\n') ;

