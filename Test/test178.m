% function test178
%TEST178 matrix realloc

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

fprintf ('test178: --------------------------------- matrix realloc\n') ;

n = 20 ;

rng ('default') ;

desc = struct ('mask', 'complement') ;

for trial = 1:10

    Cin = GB_spec_random (n, n, inf, 1, 'double') ;
    Cin.sparsity = 2 ; % sparse
    M = sparse (n,n) ;
    M (1,1) = 1 ;
    A = sparse (n,n) ;

GrB.burble (1) ;
    C1 = GB_spec_assign (Cin, M, [ ], A, [ ], [ ], desc, false) ;
    C2 = GB_mex_assign  (Cin, M, [ ], A, [ ], [ ], desc) ;
    GB_spec_compare (C1, C2) ;
    sparse (C1.matrix)
    sparse (C2.matrix)
GrB.burble (0) ;

end


fprintf ('\ntest178: all tests passed\n') ;

