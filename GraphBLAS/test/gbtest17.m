function gbtest17
%GBTEST17 test gb.gbtranspose

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

rng ('default') ;

n = 6 ;
m = 7 ;
A = 100 * sprand (n, m, 0.5) ;
AT = A' ;
M = sparse (rand (m,n)) > 0.5 ;
Cin = sprand (m, n, 0.5) ;

Cout = gb.gbtranspose (A) ;
assert (gbtest_eq (AT, Cout)) ;

Cout = gb.gbtranspose (A) ;
assert (gbtest_eq (AT, Cout)) ;

Cout = gb.gbtranspose (Cin, M, A) ;
C2 = Cin ;
C2 (M) = AT (M) ;
assert (gbtest_eq (C2, Cout)) ;

Cout = gb.gbtranspose (Cin, '+', A) ;
C2 = Cin + AT ;
assert (gbtest_eq (C2, Cout)) ;

d.in0 = 'transpose' ;
Cout = gb.gbtranspose (Cin', M', A, d) ;
C2 = Cin' ;
C2 (M') = A (M') ;
assert (gbtest_eq (C2, Cout)) ;

Cout = gb.gbtranspose (Cin', '+', A, d) ;
C2 = Cin' + A ;
assert (gbtest_eq (C2, Cout)) ;

d.mask = 'complement' ;
d2 = d ;
d2.kind = 'sparse' ;
Cout  = gb.gbtranspose (Cin', M', A, d) ;
Cout2 = gb.gbtranspose (Cin', M', A, d2) ;
C2 = Cin' ;
C2 (~M') = A (~M') ;
assert (gbtest_eq (C2, Cout)) ;
assert (gbtest_eq (C2, Cout2)) ;
assert (isequal (class (Cout2), 'double')) ;

fprintf ('gbtest17: all tests passed\n') ;

