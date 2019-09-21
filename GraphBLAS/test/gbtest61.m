function gbtest61
%GBTEST61 test gb.laplacian

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

rng ('default') ;
n = 10 ;
A = sprand (n, n, 0.4) ;

S = tril (A, -1) ;
S = S+S' ;
G = gb (S) ;

L0 = laplacian (graph (S, 'OmitSelfLoops')) ;

L1 = gb.laplacian (S) ;
L2 = gb.laplacian (G) ;
L3 = gb.laplacian (G, 'double', 'check') ;

assert (isequal (L0, L1)) ;
assert (isequal (L0, L2)) ;
assert (isequal (L0, L3)) ;

G = gb (G, 'by row') ;

L2 = gb.laplacian (G) ;
L3 = gb.laplacian (G, 'double', 'check') ;

assert (isequal (L0, L2)) ;
assert (isequal (L0, L3)) ;

types = { 'double', 'single', 'int8', 'int16', 'int32', 'int64' } ;
for k = 1:6
    type = types {k} ;
    L2 = gb.laplacian (G, type) ;
    assert (isequal (gb.type (L2), type)) ;
    assert (isequal (L0, double (L2))) ;
end

ok = true ;
try
    L = gb.laplacian (G, 'uint8') ;
    ok = false ;
catch expected_error
    expected_error
end
assert (ok) ;

ok = true ;
try
    L = gb.laplacian (A, 'double', 'check') ;
    ok = false ;
catch expected_error
    expected_error
end
assert (ok) ;

ok = true ;
try
    L = gb.laplacian (S + speye (n), 'double', 'check') ;
    ok = false ;
catch expected_error
    expected_error
end
assert (ok) ;


fprintf ('gbtest61: all tests passed\n') ;

