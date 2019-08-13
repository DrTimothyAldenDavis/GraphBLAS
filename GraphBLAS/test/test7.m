function test7
%TEST7 test gb.build

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

n = 5 ;
A = sprand (n, n, 0.5) ;
A (n,n) = 5 ;

[i j x] = find (A) ;
[m n] = size (A) ;

G = gb.build (i, j, x, m, n)
S = sparse   (i, j, x, m, n)

sparse (G)-S
assert (isequal (S, sparse (G)))

d.kind = 'gb' ;
G = gb.build (i, j, x, m, n, d) ;
S - sparse (G)
assert (isequal (S, sparse (G)))

G = gb.build (i, j, x, m, n, d)
S - sparse (G)
assert (isequal (S, sparse (G)))

i0 = int64 (i) - 1 ;
j0 = int64 (j) - 1 ;

S
G = gb.build (i0, j0, x)
S - sparse (G)
assert (isequal (S, sparse (G)))

fprintf ('test7: all tests passed\n') ;

