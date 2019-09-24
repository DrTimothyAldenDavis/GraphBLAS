function gbtest68
%GBTEST68 test isequal

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

rng ('default') ;

s = gb (pi) ;

assert (~isequal (s, magic (2))) ;
assert (~isequal (s, [pi pi])) ;
assert (~isequal (s, sparse (0))) ;

A = gb (2,2) ;
B = gb (2,2) ;
A (1,1) = 1 ;
B (2,2) = 1 ;
assert (~isequal (A, B)) ;

assert (~isequal (gb (A, 'int8'), gb (B, 'uint8'))) ;

fprintf ('gbtest68: all tests passed\n') ;

