function test168
%TEST168 C=A+B with C and B full, A bitmap

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

rng ('default') ;

fprintf ('test168:\n') ;

n = 10 ;
C = GB_spec_random (n, n, inf, 1, 'double') ;
C.sparsity = 8 ;    % full

A = GB_spec_random (n, n, 0.5, 1, 'double') ;
A.sparsity = 4 ;    % bitmap

B = GB_spec_random (n, n, inf, 1, 'double') ;
B.sparsity = 8 ;    % full

GrB.burble (1) ;
C1 = A.matrix + B.matrix ;
C2 = GB_mex_Matrix_eWiseAdd (C, [ ], [ ], 'plus', A, B, [ ]) ;
GB_spec_compare (C1, C2) ;
GrB.burble (0) ;

fprintf ('test168: all tests passed\n') ;

