clear all
addpath ../../Test/spok

rng ('default') ;
A = 100 * rand (4) ;
A (1,1) = 0
S = sparse (A)
assert (isequal (S, sparse (full (gb (S)))))
assert (isequal (S, sparse (full (full (gb (S))))))
assert (isequal (S, sparse (full (sparse (full (gb (S)))))))

S2 = sparse (gb (full (sparse (full (gb (S))))))
spok (S2) ;
assert (norm (S-S2,1) == 0)
S2 = 1*S2 ;
spok (S2) ;
assert (isequal (S, S2))

S2 = 1*sparse (gb (sparse (gb (full (sparse (full (gb (S))))))))
assert (isequal (S, S2))

S = logical (S) ;
assert (isequal (S, sparse (full (gb (S)))))

X = int8 (A)
G = gb (X)
spok (sparse (G)) ;
assert (isequal (X, full (sparse (G))))
% X
% full (G)
assert (isequal (X, full (G)))

X = int16 (A)
G = gb (X)
spok (sparse (G)) ;
assert (isequal (X, full (sparse (G))))
assert (isequal (X, full (G)))

X = int32 (A)
G = gb (X)
spok (sparse (G)) ;
assert (isequal (X, full (sparse (G))))
assert (isequal (X, full (G)))

X = int64 (A)
G = gb (X)
spok (sparse (G)) ;
assert (isequal (X, full (sparse (G))))
assert (isequal (X, full (G)))

X = uint8 (A)
G = gb (X)
spok (sparse (G)) ;
assert (isequal (X, full (sparse (G))))
assert (isequal (X, full (G)))

X = uint16 (A)
G = gb (X)
spok (sparse (G)) ;
assert (isequal (X, full (sparse (G))))
assert (isequal (X, full (G)))

X = uint32 (A)
G = gb (X)
spok (sparse (G)) ;
assert (isequal (X, full (sparse (G))))
assert (isequal (X, full (G)))

X = uint64 (A)
G = gb (X)
spok (sparse (G)) ;
full (G)
assert (isequal (X, full (sparse (G))))
assert (isequal (X, full (G)))

fprintf ('test11: all tests passed\n') ;

