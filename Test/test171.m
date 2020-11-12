function test171
%TEST171 test conversion

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

rng ('default') ;

fprintf ('test171:\n') ;

GrB.burble (1) ;

for n = [ 1000 1025 2000 ]

    A.matrix = rand (n) ;
    A.sparsity = 4 ;
    C = GB_mex_dump (A, 1) ;

    A.matrix = sparse (rand (n)) ;
    A.sparsity = 4 ;
    C = GB_mex_dump (A, 1) ;
end

GrB.burble (0) ;
fprintf ('test171: all tests passed\n') ;

