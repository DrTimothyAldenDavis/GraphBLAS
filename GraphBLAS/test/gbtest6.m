function gbtest6
%GBTEST6 test GrB.mxm

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

rng ('default') ;
A = sparse (rand (2)) ;
B = sparse (rand (2)) ;

C = A*B ;

G = GrB.mxm ('+.*', A, B) ;
err = norm (C-G, 1) ;
assert (err < 1e-12) ;

d.kind = 'sparse' ;
d.in0 = 'transpose' ;
G = GrB.mxm ('+.*', A, B, d) ;
C = A'*B ;

err = norm (C-G, 1) ;
assert (err < 1e-12) ;

d.kind = 'GrB' ;
G = GrB.mxm ('+.*', A, B, d) ;
err = norm (C-G, 1) ;
assert (err < 1e-12) ;
clear d

E = sparse (rand (2)) ;
C = E + A*B ;
G = GrB.mxm (E, '+', '+.*', A, B) ;
err = norm (C-G, 1) ;
assert (err < 1e-12) ;

M = false (2,2) ;
Cin = rand (2) ;
M (1,1) = 1 ;
G = GrB.mxm (Cin, M, '+', '+.*', A, B) ;
T = Cin + A*B ;
C = Cin ;
C (M) = T (M) ;
err = norm (C-G, 1) ;
assert (err < 1e-12)

n = 10 ;
A = sprand (n, n, 0.1) ;
B = rand (n) ;
G = GrB.mxm ('+.*', A, B) ;
E = GrB (A) * B ;
C = A*B ;
err = norm (C-G, 1) ;
err = norm (E-G, 1) ;
assert (err < 1e-12) ;

% G is exported as a MATLAB/Octave sparse matrix, but as double-complex instead
% of single-complex, since MATLAB/Octave do not yet have sparse single complex
% matrices (at least earlier versions of those packages).
clear d
d.kind = 'builtin' ;
A = A + 1i * sprand (n, n, 0.1) ;
A = GrB (A, 'single complex') ;
G = GrB.mxm ('+.*', A, A, d) ;
B = complex (A) ;
C = B*B ;
err = norm (C-G, 1) ;
assert (err < 1e-6) ;
assert (isequal (GrB.type (G), 'double complex')) ;
[f,s] = GrB.format (G) ;
assert (isequal (s, 'sparse')) ;

% full matrices can be exported as single complex MATLAB/Octave matrices
d.kind = 'full' ;
G = GrB.mxm ('+.*', A, A, d) ;
assert (isequal (GrB.type (G), 'single complex')) ;
[f,s] = GrB.format (G) ;
assert (isequal (s, 'full')) ;

fprintf ('gbtest6: all tests passed\n') ;

