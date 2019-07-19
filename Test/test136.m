function test136
%TEST136 GxB_subassign, C(I,J)<M> += A, using method 08

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

fprintf ('test136: GxB_subassign, C(I,J)<M> += A, using method 08\n') ;

rng ('default') ;

m = 1000 ;
n = 5 ;
am = 500 ;
an = n ;

C = sprand (m, n, 0.1) ;
I = randperm (m, am) ;

M = spones (sprand (am, an, 0.1)) ;
A = sprand (am, an, 0.1) ;
I0 = uint64 (I) - 1 ;

M (:,1) = 0 ;
M (1:2,1) = 1 ;
A (:,1) = sprand (am, 1, 0.8)  ;

A (:,2) = 0 ;
A (1:2,2) = 1 ;
M (:,2) = spones (sprand (am, 1, 0.8))  ;

C2 = GB_mex_subassign (C, M, 'plus', A, I0, [ ]) ;

fprintf ('test136: all tests passed\n') ;
