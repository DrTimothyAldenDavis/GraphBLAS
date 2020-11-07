function test84
%TEST84 test GrB_assign (row and column with C in CSR/CSC format)

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

fprintf ('\ntest84: GrB_assign with row/col assignments\n') ;

rng ('default') ;
m = 10 ;
n = 20 ;

% create a CSR matrix
C0 = GB_spec_random (m, n, 0.5, 100, 'double', false, false) ;

Mcol.matrix = sparse (ones (m,1)) ; % spones (sprandn (m, 1, 0.5)) ;
Mcol.sparsity = 8 ;
Mrow.matrix = sparse (ones (n,1)) ; % spones (sprandn (n, 1, 0.5)) ;
Mrow.sparsity = 8 ;

Acol = sprandn (4, 1, 0.5)  ;
Arow = sprandn (4, 1, 0.5)  ;

J = [3 4 5 6] ;
J0 = uint64 (J) - 1 ;
I = 2 ;
I0 = uint64 (I) - 1 ;

for sparsity_control = 1:15
    fprintf ('.') ;
    C0.sparsity = sparsity_control ;
    for csc = 0:1
        C0.is_csc = csc ;

        % row assign
        C1 = GB_mex_assign      (C0, Mrow, 'plus', Arow, I0, J0, [ ], 2) ;
        C2 = GB_spec_Row_assign (C0, Mrow, 'plus', Arow, I,  J,  [ ]) ;
        GB_spec_compare (C1, C2) ;

        % col assign
        C1 = GB_mex_assign      (C0, Mcol, 'plus', Acol, J0, I0, [ ], 1) ;
        C2 = GB_spec_Col_assign (C0, Mcol, 'plus', Acol, J,  I,  [ ]) ;
        GB_spec_compare (C1, C2) ;

        % row assign, no accum
        C1 = GB_mex_assign      (C0, Mrow, [ ], Arow, I0, J0, [ ], 2) ;
        C2 = GB_spec_Row_assign (C0, Mrow, [ ], Arow, I,  J,  [ ]) ;
        GB_spec_compare (C1, C2) ;

        % col assign, no accum
        C1 = GB_mex_assign      (C0, Mcol, [ ], Acol, J0, I0, [ ], 1) ;
        C2 = GB_spec_Col_assign (C0, Mcol, [ ], Acol, J,  I,  [ ]) ;
        GB_spec_compare (C1, C2) ;

    end
end

fprintf ('\ntest84: all tests passed\n') ;

