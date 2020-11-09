function test180
%TEST180 subassign

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

fprintf ('test180: --------------------------------- assign\n') ;

n = 20 ;
rng ('default') ;

Cin = GB_spec_random (n, n, 0.5, 1, 'double') ;
I1 = [2 3 5 1 9 11] ;
J1 = [4 5 1 9 2 12] ;
I0 = uint64 (I1) - 1 ;
J0 = uint64 (J1) - 1 ;
M1 = logical (sprand (m, m, 0.5)) ;
A = GB_spec_random (m, m, 0.5, 1, 'double') ;

for c = 1:15
    Cin.sparsity = c ;
    for a = 1:15
        A.sparsity = a ;
        C1 = GB_spec_subassign (Cin, [ ], [ ], A, I1, J1, [ ]) ;
        C2 = GB_mex_subassign  (Cin, [ ], [ ], A, I1, J1, [ ]) ;
        GB_spec_compare (C1, C2) ;
    end
end

fprintf ('\ntest180: all tests passed\n') ;

