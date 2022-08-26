function gbtest77
%GBTEST77 test error handling
% All errors generated by this test are expected.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2022, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

rng ('default') ;
ok = true ;
A = magic (5) ;
G = GrB (A) ;
Z = GrB (rand (3) + 1i*rand(3)) ;
M = G > 10 ;
I = GrB (int64 (magic (4))) ;
R = rand (3,4) ;

try
    C = gammaln (Z) %#ok<*NASGU>
    ok = false ;
catch expected_error
    expected_error %#ok<*NOPRT>
    disp (expected_error.stack (end-1))
end
assert (ok) ;

try
    C = gamma (Z)
    ok = false ;
catch expected_error
    expected_error
    disp (expected_error.stack (end-1))
end
assert (ok) ;

try
    C = erf (Z)
    ok = false ;
catch expected_error
    expected_error
    disp (expected_error.stack (end-1))
end
assert (ok) ;

try
    C = erfc (Z)
    ok = false ;
catch expected_error
    expected_error
    disp (expected_error.stack (end-1))
end
assert (ok) ;

try
    C = cbrt (Z)
    ok = false ;
catch expected_error
    expected_error
    disp (expected_error.stack (end-1))
end
assert (ok) ;

try
    C = atan2 (Z,G)
    ok = false ;
catch expected_error
    expected_error
    disp (expected_error.stack (end-1))
end
assert (ok) ;

try
    C = bitcmp (Z)
    ok = false ;
catch expected_error
    expected_error
    disp (expected_error.stack (end-1))
end
assert (ok) ;

try
    C = bitcmp (M)
    ok = false ;
catch expected_error
    expected_error
    disp (expected_error.stack (end-1))
end
assert (ok) ;

try
    C = bitcmp (I, 'double')
    ok = false ;
catch expected_error
    expected_error
    disp (expected_error.stack (end-1))
end
assert (ok) ;

try
    C = bitget (Z, 1)
    ok = false ;
catch expected_error
    expected_error
    disp (expected_error.stack (end-1))
end
assert (ok) ;

try
    C = bitget (M, 1)
    ok = false ;
catch expected_error
    expected_error
    disp (expected_error.stack (end-1))
end
assert (ok) ;

try
    C = bitget (I, 1, 'double')
    ok = false ;
catch expected_error
    expected_error
    disp (expected_error.stack (end-1))
end
assert (ok) ;

try
    C = bitset (Z, 1, 0)
    ok = false ;
catch expected_error
    expected_error
    disp (expected_error.stack (end-1))
end
assert (ok) ;

try
    C = bitset (M, 1, 0)
    ok = false ;
catch expected_error
    expected_error
    disp (expected_error.stack (end-1))
end
assert (ok) ;

try
    C = bitset (I, 1, 0, 'double')
    ok = false ;
catch expected_error
    expected_error
    disp (expected_error.stack (end-1))
end
assert (ok) ;

try
    C = complex (Z, Z)
    ok = false ;
catch expected_error
    expected_error
    disp (expected_error.stack (end-1))
end
assert (ok) ;

try
    C = bitand (Z, Z)
    ok = false ;
catch expected_error
    expected_error
    s = expected_error.stack ;
    for k = 1:length (s)
        disp (s (k)) ;
    end
end
assert (ok) ;

try
    C = bitand (M, M)
    ok = false ;
catch expected_error
    expected_error
    s = expected_error.stack ;
    for k = 1:length (s)
        disp (s (k)) ;
    end
end
assert (ok) ;

try
    C = bitand (G, G, 'double')
    ok = false ;
catch expected_error
    expected_error
    s = expected_error.stack ;
    for k = 1:length (s)
        disp (s (k)) ;
    end
end
assert (ok) ;

try
    C = bitand (GrB (G, 'int8'), GrB (G, 'int16'))
    ok = false ;
catch expected_error
    expected_error
    s = expected_error.stack ;
    for k = 1:length (s)
        disp (s (k)) ;
    end
end
assert (ok) ;

try
    I = GrB (1.5) ;
    C = G (I,I)
    ok = false ;
catch expected_error
    expected_error
    s = expected_error.stack ;
    for k = 1:length (s)
        disp (s (k)) ;
    end
end
assert (ok) ;

try
    I = GrB (1.5, 'double complex') ;
    C = G (I,I)
    ok = false ;
catch expected_error
    expected_error
    s = expected_error.stack ;
    for k = 1:length (s)
        disp (s (k)) ;
    end
end
assert (ok) ;

try
    C = GrB.empty ([1 2 3 0]) ;
    ok = false ;
catch expected_error
    expected_error
    s = expected_error.stack ;
    for k = 1:length (s)
        disp (s (k)) ;
    end
end
assert (ok) ;

try
    C = G (1, { 1,2,3,4 }) ;
    ok = false ;
catch expected_error
    expected_error
    s = expected_error.stack ;
    for k = 1:length (s)
        disp (s (k)) ;
    end
end
assert (ok) ;

try
    G ([1 2 3]) = 52 
    ok = false ;
catch expected_error
    expected_error
    s = expected_error.stack ;
    for k = 1:length (s)
        disp (s (k)) ;
    end
end
assert (ok) ;

try
    issymmetric (G, 'crud') 
    ok = false ;
catch expected_error
    expected_error
    s = expected_error.stack ;
    for k = 1:length (s)
        disp (s (k)) ;
    end
end
assert (ok) ;

try
    max (G, [ ], 42) ;
    ok = false ;
catch expected_error
    expected_error
    s = expected_error.stack ;
    for k = 1:length (s)
        disp (s (k)) ;
    end
end
assert (ok) ;

try
    min (G, [ ], 42) ;
    ok = false ;
catch expected_error
    expected_error
    s = expected_error.stack ;
    for k = 1:length (s)
        disp (s (k)) ;
    end
end
assert (ok) ;

try
    GrB.random (4, 5, 0.6, 'range', [1 2 3])
    ok = false ;
catch expected_error
    expected_error
    s = expected_error.stack ;
    for k = 1:length (s)
        disp (s (k)) ;
    end
end
assert (ok) ;

try
    A (GrB (1.3))
    ok = false ;
catch expected_error
    expected_error
    s = expected_error.stack ;
    for k = 1:length (s)
        disp (s (k)) ;
    end
end
assert (ok) ;

try
    A (GrB (1, 'complex'))
    ok = false ;
catch expected_error
    expected_error
    s = expected_error.stack ;
    for k = 1:length (s)
        disp (s (k)) ;
    end
end
assert (ok) ;

try
    A (GrB (true))
    ok = false ;
catch expected_error
    expected_error
    s = expected_error.stack ;
    for k = 1:length (s)
        disp (s (k)) ;
    end
end
assert (ok) ;

try
    sprand (G, 0)
    ok = false ;
catch expected_error
    expected_error
    s = expected_error.stack ;
    for k = 1:length (s)
        disp (s (k)) ;
    end
end
assert (ok) ;

try
    sprandn (G, 0)
    ok = false ;
catch expected_error
    expected_error
    s = expected_error.stack ;
    for k = 1:length (s)
        disp (s (k)) ;
    end
end
assert (ok) ;

try
    max (Z)
    ok = false ;
catch expected_error
    expected_error
    s = expected_error.stack ;
    for k = 1:length (s)
        disp (s (k)) ;
    end
end
assert (ok) ;

try
    min (Z)
    ok = false ;
catch expected_error
    expected_error
    s = expected_error.stack ;
    for k = 1:length (s)
        disp (s (k)) ;
    end
end
assert (ok) ;

try
    max (G, G, 2)
    ok = false ;
catch expected_error
    expected_error
    s = expected_error.stack ;
    for k = 1:length (s)
        disp (s (k)) ;
    end
end
assert (ok) ;

try
    min (G, G, 2)
    ok = false ;
catch expected_error
    expected_error
    s = expected_error.stack ;
    for k = 1:length (s)
        disp (s (k)) ;
    end
end
assert (ok) ;

try
    L = GrB.laplacian (R)
    ok = false ;
catch expected_error
    expected_error
    s = expected_error.stack ;
    for k = 1:length (s)
        disp (s (k)) ;
    end
end
assert (ok) ;

try
    K = GrB.incidence (G, 'uint8')
    ok = false ;
catch expected_error
    expected_error
    s = expected_error.stack ;
    for k = 1:length (s)
        disp (s (k)) ;
    end
end
assert (ok) ;

try
    GrB.expand (rand (3), rand (4)) ;
    ok = false ;
catch expected_error
    expected_error
    s = expected_error.stack ;
    for k = 1:length (s)
        disp (s (k)) ;
    end
end
assert (ok) ;

try
    GrB.apply (1) ;
    ok = false ;
catch expected_error
    expected_error
    s = expected_error.stack ;
    for k = 1:length (s)
        disp (s (k)) ;
    end
end
assert (ok) ;

try
    opts.type = 'int32' ;
    GrB.pagerank (G, opts) ;
    ok = false ;
catch expected_error
    expected_error
    s = expected_error.stack ;
    for k = 1:length (s)
        disp (s (k)) ;
    end
end
assert (ok) ;

try
    norm (G, 3)
    ok = false ;
catch expected_error
    expected_error
    s = expected_error.stack ;
    for k = 1:length (s)
        disp (s (k)) ;
    end
end
assert (ok) ;

try
    C = GrB.apply2 (G, '', '', pi, G) ;
    ok = false ;
catch expected_error
    expected_error
    s = expected_error.stack ;
    for k = 1:length (s)
        disp (s (k)) ;
    end
end
assert (ok) ;

try
    C = GrB.select ('garbage', G) ;
    ok = false ;
catch expected_error
    expected_error
    s = expected_error.stack ;
    for k = 1:length (s)
        disp (s (k)) ;
    end
end
assert (ok) ;

try
    C = GrB.eunion (G, G, G, G, G, G, G) ;
    ok = false ;
catch expected_error
    expected_error
    s = expected_error.stack ;
    for k = 1:length (s)
        disp (s (k)) ;
    end
end
assert (ok) ;

try
    blob = GrB.serialize (G, 'garbage') ;
    ok = false ;
catch expected_error
    expected_error
    s = expected_error.stack ;
    for k = 1:length (s)
        disp (s (k)) ;
    end
end
assert (ok) ;

try
    I = [1 2] ;
    J = [3 4] ;
    X = [pi 2] ;
    gunk = magic (3) ;
    C = GrB.build (I, J, X, gunk, gunk) ;
    ok = false ;
catch expected_error
    expected_error
    s = expected_error.stack ;
    for k = 1:length (s)
        disp (s (k)) ;
    end
end
assert (ok) ;

try
    [C,P] = GrB.argsort (G, 2, 'garbage') ;
    ok = false ;
catch expected_error
    expected_error
    s = expected_error.stack ;
    for k = 1:length (s)
        disp (s (k)) ;
    end
end
assert (ok) ;

try
    [f,s,iso] = GrB.format ;
    ok = false ;
catch expected_error
    expected_error
    s = expected_error.stack ;
    for k = 1:length (s)
        disp (s (k)) ;
    end
end
assert (ok) ;

fprintf ('gbtest77: all tests passed\n') ;

