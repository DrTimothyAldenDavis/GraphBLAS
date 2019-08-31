function gbtest12
%GBTEST12 test gb.eadd, gb.emult

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

rng ('default') ;
A = sparse (rand (2)) ;
B = sparse (rand (2)) ;

C = A+B ;
D = A.*B ;

G = gb.eadd ('+', A, B) ;
err = norm (C-G, 1)
assert (logical (err < 1e-12))

H = gb.emult ('*', A, B) ;
err = norm (D-H, 1)
assert (logical (err < 1e-12))

d.kind = 'sparse' ;
d.in0 = 'transpose' ;
d

G = gb.eadd ('+', A, B, d) ;
C = A'+B ;
err = norm (C-G, 1)
assert (logical (err < 1e-12))

H = gb.emult ('*', A, B, d) ;
D = A'.*B ;
err = norm (H-D, 1)
assert (logical (err < 1e-12))

d.kind = 'gb' ;
G = gb.eadd ('+', A, B, d) ;
err = norm (C-G, 1)

H = gb.emult ('*', A, B, d) ;
err = norm (D-H, 1)

E = sparse (rand (2)) ;
C = E + A+B ;
G = gb.eadd (E, '+', '+', A, B) ;
C-G

F = sparse (rand (2)) ;
D = F + A.*B ;
H = gb.emult (F, '+', '*', A, B) ;
D-H
assert (gbtest_eq (D, H)) ;

G = gb.eadd ('+', A, B)
C = A+B
assert (gbtest_eq (C, G)) ;

H = gb.emult ('*', A, B)
D = A.*B
assert (gbtest_eq (D, H)) ;

fprintf ('gbtest12: all tests passed\n') ;

