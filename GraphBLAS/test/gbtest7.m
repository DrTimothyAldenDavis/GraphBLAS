function gbtest7
%GBTEST7 test gb.build

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

n = 5 ;
A = sprand (n, n, 0.5) ;
A (n,n) = 5 ;

[i j x] = find (A) ;
[m n] = size (A) ;

G = gb.build (i, j, x, m, n) ;
S = sparse   (i, j, x, m, n) ;
assert (gbtest_eq (S, G)) ;

d.kind = 'gb' ;
G = gb.build (i, j, x, m, n, d) ;
assert (gbtest_eq (S, G)) ;

d.kind = 'sparse' ;
G = gb.build (i, j, x, m, n, d) ;
assert (gbtest_eq (S, G))

i0 = int64 (i) - 1 ;
j0 = int64 (j) - 1 ;

G = gb.build (i0, j0, x) ;
assert (gbtest_eq (S, G)) ;

G = gb.build (1:3, 1:3, [1 1 1]) ;
assert (gbtest_eq (speye (3), G)) ;


fprintf ('gbtest7: all tests passed\n') ;

