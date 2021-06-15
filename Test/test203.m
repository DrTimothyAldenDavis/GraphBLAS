function test203
%TEST203 test iso subref

% GrB.burble (1) ;
n = 10 ;
A.matrix = pi * spones (sprandn (n, n, 0.5)) ;
A.class = 'double' ;
A.iso = true ;

m = 5 ;
I = 1:m ;
I0 = uint64 (I) - 1 ;

Cin = sparse (m,m) ;
C1 = GB_mex_Matrix_extract (Cin, [ ], [ ], A, I0, I0, [ ]) ;
C2 = A.matrix (I,I) ;
assert (isequal (C1.matrix, C2)) ;

% GrB.burble (0) ;
fprintf ('test203: all tests passed\n') ;

