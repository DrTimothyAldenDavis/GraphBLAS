function gbtest14
%GBTEST14 test gb.gbkron

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

rng ('default') ;
A = sparse (rand (2,3)) ;
B = sparse (rand (4,8)) ;

C = kron (A,B) ;

G = gb.gbkron ('*', A, B) ;
err = norm (C-G, 1)
assert (logical (err < 1e-12))

d.kind = 'sparse' ;
d.in0 = 'transpose' ;

G = gb.gbkron ('*', A, B, d) ;
C = kron (A', B) ;
err = norm (C-G, 1)
assert (logical (err < 1e-12))

d.kind = 'gb' ;
G = gb.gbkron ('*', A, B, d) ;
err = norm (C-G, 1)

E = sparse (rand (8,24)) ;
C = E + kron (A,B) ;
G = gb.gbkron (E, '+', '*', A, B) ;
C-G

fprintf ('gbtest14: all tests passed\n') ;

