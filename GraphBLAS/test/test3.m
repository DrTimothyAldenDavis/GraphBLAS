function test3
%TEST3 test gb.build

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

rng ('default') ;
m = 10 ;
n = 5 ;
A = sprand (m, n, 0.5) ;

[i j x] = find (A) ;

C = gb.build (i, j, x, m, n) ;

S = sparse (C) ;
assert (isequal (S, A)) ;

% Prob = ssget (2662)
% A = Prob.A ;
fprintf ('Generating large test matrix; please wait ...\n') ;
% n = 1e6 ;
% nz = 50e6 ;
n = 1000 ;
nz = 7000 ;
density = nz / n^2 ;
tic
A = sprand (n, n, density) ;
B = sprand (n, n, density) ;
A = kron (A,B) ;
clear B
t = toc ;
fprintf ('%12.4f sec : A = sprand(n,n,nz), n: %g nz %g\n', t, n, nnz (A)) ;

[i j x] = find (A) ;
[m n] = size (A) ;

i0 = uint64 (i) - 1 ;
j0 = uint64 (j) - 1 ;

nthreads = gb.threads ;
fprintf ('using %d threads in GraphBLAS\n', nthreads) ;

fprintf ('\nwith [I J] already sorted on input:\n') ;

tic
A1 = sparse (i, j, x, m, n) ;
t = toc ;
fprintf ('%12.4f sec : A = sparse (i, j, x, m, n) ;\n', t) ;

tic
A3 = gb.build (i, j, x, m, n) ;
t = toc ;
fprintf ('%12.4f sec : A = gb.build (i, j, x, m, n), same inputs as MATLAB\n', t) ;

tic
A2 = gb.build (i0, j0, x, m, n) ;
t = toc ;
fprintf ('%12.4f sec : A = gb.build (i0, j0, x, m, n), with i0 and j0 uint64\n', t) ;

A2 = sparse (A2) ;
A3 = sparse (A3) ;
assert (isequal (A1, sparse (A2))) ;
assert (isequal (A1, sparse (A3))) ;

fprintf ('\nwith [I J] jumbled so that a sort is required:\n') ;

i = i (end:-1:1) ;
j = j (end:-1:1) ;
i (1:10) = i (randperm (10)) ;
i0 = uint64 (i) - 1 ;
j0 = uint64 (j) - 1 ;

tic
A1 = sparse (i, j, x, m, n) ;
t = toc ;
fprintf ('%12.4f sec : A = sparse (i, j, x, m, n) ;\n', t) ;

tic
A3 = gb.build (i, j, x, m, n) ;
t = toc ;
fprintf ('%12.4f sec : A = gb.build (i, j, x, m, n), same inputs as MATLAB\n', t) ;

tic
A2 = gb.build (i0, j0, x, m, n) ;
t = toc ;
fprintf ('%12.4f sec : A = gb.build (i0, j0, x, m, n), with i0 and j0 uint64\n', t) ;

tic
A2 = sparse (A2) ;
t = toc ;
fprintf ('%12.4f sec : A = gb.sparse (A) to convert from GraphBLAS to MATLAB\n', t);

assert (isequal (A1, A2)) ;

fprintf ('\ntest3: all tests passed\n') ;

