function gbtest59
%GBTEST59 test end

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

A = rand (4,7) ;
G = gb (A) ;

A = A (2:end, 3:end) ;
G = G (2:end, 3:end) ;
assert (isequal (G, A)) ;

A = A (2:2:end, 3:2:end) ;
G = G (2:2:end, 3:2:end) ;
assert (isequal (G, A)) ;

A = rand (7, 1) ;
G = gb (A) ;

A = A (2:2:end) ;
G = G (2:2:end) ;
assert (isequal (G, A)) ;

ok = true ;
G = gb (magic (2)) ;
try
    G = G (2:end)
    ok = false ;
catch expected_error
    expected_error
end
assert (ok)

ok = true ;
try
    G = G (2:end, 1:end, 1:end)
    ok = false ;
catch expected_error
    expected_error
end
assert (ok)

fprintf ('gbtest59: all tests passed\n') ;

