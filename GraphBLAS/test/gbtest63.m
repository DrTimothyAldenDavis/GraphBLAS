function gbtest63
%GBTEST63 test gb.ktruss and gb.tricount

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

rng ('default') ;

load west0479 ;
A = gb.offdiag (west0479) ;
A = A+A' ;
C3a  = gb.ktruss (A) ;
C3  = gb.ktruss (A, 3) ;
assert (isequal (C3a, C3)) ;
C3  = gb.ktruss (A, 3, 'check') ;
assert (isequal (C3a, C3)) ;

ntriangles = sum (C3, 'all') / 6 ;

C4a = gb.ktruss (A, 4) ;
C4b = gb.ktruss (C3, 4) ;          % this is faster
assert (isequal (C4a, C4b)) ;

nt2 = gb.tricount (A) ;
assert (ntriangles == nt2) ;
assert (ntriangles == 235) ;

ok = true ;
try
    C = gb.ktruss (A, 2) ;
    ok = false ;
catch expected_error
    expected_error
end
assert (ok) ;

ok = true ;
try
    C = gb.ktruss (rand (3,4)) ;
    ok = false ;
catch expected_error
    expected_error
end
assert (ok) ;

A = west0479 ;
ok = true ;
try
    C = gb.ktruss (A, 3, 'check') ;
    ok = false ;
catch expected_error
    expected_error
end
assert (ok) ;

A = A+A' ;
ok = true ;
try
    C = gb.ktruss (A, 3, 'check') ;
    ok = false ;
catch expected_error
    expected_error
end
assert (ok) ;

fprintf ('gbtest63: all tests passed\n') ;

