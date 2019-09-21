function gbtest54
%GBTEST54 test gb.compact

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

rng ('default') ;
n = 32 ;
H = gb (n,n) ;
I = sort (randperm (n, 4)) ;
J = sort (randperm (n, 4)) ;
A = magic (4) ;
H (I,J) = A ;
[C, I, J] = gb.compact (H) ;
H (I, J(1)) = 0 ;
[C, I, J] = gb.compact (H, 0) ;
assert (isequal (C, A (:,2:end))) ;

A = ones (4) ;
A (1,1) = 2 ;
G = gb.compact (A, 2) ;
assert (nnz (G) == 15) ;
A = ones (4) ;
A (1,1) = 0 ;
A = sparse (A) ;
assert (isequal (G, A)) ;

fprintf ('gbtest54: all tests passed\n') ;

