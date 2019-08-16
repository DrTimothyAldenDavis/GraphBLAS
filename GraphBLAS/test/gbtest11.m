function gbtest11
%GBTEST11 test gb, sparse

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

rng ('default') ;
A = 100 * rand (4) ;
A (1,1) = 0
S = sparse (A)
assert (isequal (S, double (full (gb (S)))))
assert (isequal (S, double (full (full (gb (S))))))
assert (isequal (S, double (full (double (full (gb (S)))))))

S2 = double (gb (full (double (full (gb (S))))))
assert (norm (S-S2,1) == 0)
% S2 = 1*S2 ;
assert (isequal (S, S2))

S2 = double (gb (double (gb (full (double (full (gb (S))))))))
assert (isequal (S, S2))

S = logical (S) ;
assert (isequal (S, double (full (gb (S)))))

X = int8 (A)
G = gb (X)
assert (isequal (X, full (double (G))))
assert (isequal (X, double (full (G))))

X = int16 (A)
G = gb (X)
assert (isequal (X, full (double (G))))
assert (isequal (X, double (full (G))))

X = int32 (A)
G = gb (X)
assert (isequal (X, full (double (G))))
assert (isequal (X, double (full (G))))

X = int64 (A)
G = gb (X)
assert (isequal (X, full (double (G))))
assert (isequal (X, double (full (G))))

X = uint8 (A)
G = gb (X)
assert (isequal (X, full (double (G))))
assert (isequal (X, double (full (G))))

X = uint16 (A)
G = gb (X)
assert (isequal (X, full (double (G))))
assert (isequal (X, double (full (G))))

X = uint32 (A)
G = gb (X)
assert (isequal (X, full (double (G))))
assert (isequal (X, double (full (G))))

X = uint64 (A)
G = gb (X)
full (G)
assert (isequal (X, full (double (G))))
assert (isequal (X, double (full (G))))

fprintf ('gbtest11: all tests passed\n') ;

