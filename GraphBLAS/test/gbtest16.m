function gbtest16
%GBTEST16 test gb.extract

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

rng ('default') ;

n = 6 ;
A = 100 * sprand (n, n, 0.5) ;
AT = A' ;
M = sparse (rand (n)) > 0.5 ;
Cin = sprand (n, n, 0.5) ;

Cout = gb.extract (Cin, A) ;
assert (gbtest_eq (A, Cout)) ;

Cout = gb.extract (Cin, A, { }, { }) ;
assert (gbtest_eq (A, Cout)) ;

Cout = gb.extract (A, {n, -1, 1}, {n, -1, 1}) ;
assert (gbtest_eq (A (n:-1:1, n:-1:1), Cout)) ;

Cout = gb.extract (Cin, M, A) ;
C2 = Cin ;
C2 (M) = A (M) ;
assert (gbtest_eq (C2, Cout)) ;

Cout = gb.extract (Cin, '+', A) ;
C2 = Cin + A ;
assert (gbtest_eq (C2, Cout)) ;

d.in0 = 'transpose' ;
Cout = gb.extract (Cin, M, A, d) ;
C2 = Cin ;
C2 (M) = AT (M) ;
assert (gbtest_eq (C2, Cout)) ;

Cout = gb.extract (Cin, '+', A, d) ;
C2 = Cin + AT ;
assert (gbtest_eq (C2, Cout)) ;

d.mask = 'complement' ;
d2 = d ;
d2.kind = 'sparse' ;
Cout  = gb.extract (Cin, M, A, d) ;
Cout2 = gb.extract (Cin, M, A, d2) ;
C2 = Cin ;
C2 (~M) = AT (~M) ;
assert (gbtest_eq (C2, Cout)) ;
assert (gbtest_eq (C2, Cout2)) ;
assert (isequal (class (Cout2), 'double')) ;

I = [2 1 5] ;
J = [3 3 1 2] ;
% B = sprandn (length (I), length (J), 0.5) ;
Cout = gb.extract (A, {I}, {J}) ;
C2 = A (I,J)  ;
assert (gbtest_eq (C2, Cout)) ;

fprintf ('gbtest16: all tests passed\n') ;

