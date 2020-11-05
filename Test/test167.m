function test167
%TEST167 test C<M>=A*B with very sparse M, different types of A and B

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

semiring.add = 'plus' ;
semiring.multiply = 'times' ;
semiring.class = 'double' ;

rng ('default') ;

d = 0.02 ;
n = 1000 ;

A.matrix = 100 * sprand (n, n, d) ;
A.matrix (1:257,1) = rand (257, 1) ;

B.matrix = 100 * sprand (n, n, d) ;
B.matrix (1,1) = 1 ;
M = logical (sprand (n, n, 0.002)) ;
Cin = sparse (n, n) ;

[~, ~, ~, types, ~, ~,] = GB_spec_opsall ;
types = types.all ;

for k = 1:length (types)

    A.class = types {k} ;
    B.class = types {k} ;
    fprintf ('%s ', types {k}) ;

    A2 = GB_spec_matrix (A) ;
    B2 = GB_spec_matrix (B) ;
    % GB_spec_mxm is too slow, using built-in MATLAB semiring instead
    C1 = double (M) .* (double (A2.matrix) * double (B2.matrix)) ;
    C2 = GB_mex_mxm (Cin, M, [ ], semiring, A, B, [ ]) ;
    err = norm (C1 - C2.matrix, 1) ;
    assert (err < 1e-6) ;
end

fprintf ('\ntest167: all tests passed\n') ;

