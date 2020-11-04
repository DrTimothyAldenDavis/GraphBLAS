function test161
%TEST161 C=A*B*E

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

rng ('default') ;

n = 100 ;
d = 0.05 ;
semiring.add = 'plus' ;
semiring.multiply = 'times' ;
semiring.class = 'double' ;
GrB.burble (1) ;

for trial = 1:10
    
    A = sprand (n, n, d) ;
    B = sprand (n, n, d) ;
    E = sprand (n, n, d) ;

    C1 = A*B*E ;
    C2 = GB_mex_triple_mxm (semiring, A, B, E) ;
    GB_spec_compare (C1, C2) ;
end
