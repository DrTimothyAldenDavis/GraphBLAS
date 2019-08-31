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
Cout = gb.extract (Cin, M, A, d) ;
C2 = Cin ;
C2 (~M) = AT (~M) ;
assert (gbtest_eq (C2, Cout)) ;

I = [2 1 5] ;
J = [3 3 1 2] ;
% B = sprandn (length (I), length (J), 0.5) ;
Cout = gb.extract (A, {I}, {J}) ;
C2 = A (I,J)  ;
assert (gbtest_eq (C2, Cout)) ;

fprintf ('gbtest16: all tests passed\n') ;

