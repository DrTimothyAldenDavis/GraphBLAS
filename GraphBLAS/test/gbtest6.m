function gbtest6
%GBTEST6 test gb.mxm

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

rng ('default') ;
A = sparse (rand (2)) ;
B = sparse (rand (2)) ;

C = A*B ;

G = gb.mxm ('+.*', A, B) ;
err = norm (C-G, 1)
assert (logical (err < 1e-12))

d.kind = 'sparse' ;
d.in0 = 'transpose' ;
d
G = gb.mxm ('+.*', A, B, d) ;
C = A'*B ;

err = norm (C-G, 1)
assert (logical (err < 1e-12))

d.kind = 'gb' ;
G = gb.mxm ('+.*', A, B, d) ;
err = norm (C-G, 1)
assert (logical (err < 1e-12))

E = sparse (rand (2)) ;
C = E + A*B ;
G = gb.mxm (E, '+', '+.*', A, B)
err = norm (C-G, 1)
assert (logical (err < 1e-12))

fprintf ('gbtest6: all tests passed\n') ;

