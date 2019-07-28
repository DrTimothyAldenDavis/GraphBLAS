
clear all
gbmake

m = 10 ;
n = 5 ;
A = sprand (m, n, 0.5) ;

[i j x] = find (A) ;

C = gbbuild (i, j, x, m, n) ;

S = gbsparse (C) ;
assert (isequal (S, A)) ;

% gbthreads (4) ;
Prob = ssget (2662)
A = Prob.A ;
[i j x] = find (A) ;
[m n] = size (A) ;

i0 = uint64 (i) - 1 ;
j0 = uint64 (j) - 1 ;

tic
A1 = sparse (i, j, x, m, n) ;
toc

tic
A2 = gbbuild (i0, j0, x, m, n) ;
toc

tic
A3 = gbbuild (i, j, x, m, n) ;
toc

A2 = gbsparse (A2) ;
A3 = gbsparse (A3) ;
assert (isequal (A1, A2)) ;
assert (isequal (A1, A3)) ;

fprintf ('\njumble things\n') ;

rng ('default') ;
i = i (end:-1:1) ;
j = j (end:-1:1) ;
i (1:10) = i (randperm (10)) ;
i0 = uint64 (i) - 1 ;
j0 = uint64 (j) - 1 ;

tic
A1 = sparse (i, j, x, m, n) ;
toc

tic
A2 = gbbuild (i0, j0, x, m, n) ;
toc

tic
A2 = gbsparse (A2) ;
t = toc ;
assert (isequal (A1, A2)) ;


fprintf ('test3 passed\n') ;
fprintf ('conversion time %g\n', t) ;
