function [x, y] = gb_get_2scalars (A)
%GB_GET_PAIR get a two scalars from a parameter of length 2

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

ghb = 1 ;     % 0 for GrB, 1 for GhB

type = gbmex_type (A) ;
desc.kind = 'full' ;
a = GrB (gbmex_full (ghb, A, type, 0, desc)) ;
C = gbmex_builtin (a) ;   % export as a full MATLAB/Octave matrix
x = C (1) ;
y = C (2) ;


