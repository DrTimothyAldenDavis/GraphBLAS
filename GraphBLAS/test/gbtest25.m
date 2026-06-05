function gbtest25
%GBTEST25 test diag, tril, triu

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

rng ('default') ;
for trials = 1:10
    fprintf ('.') ;

    for m = 2:6
        for n = 2:6
            A = sprand (m, n, 0.5) ;
            G = GrB (A) ;
            for k = -m:n
                B = diag (A, k) ;
                C = diag (G, k) ;
                assert (gbtest_eq (B, C)) ;
                B = tril (A, k) ;
                C = tril (A, k) ;
                assert (gbtest_eq (B, C)) ;
                B = triu (A, k) ;
                C = triu (A, k) ;
                assert (gbtest_eq (B, C)) ;
            end
            B = diag (A) ;
            C = diag (G) ;
            assert (gbtest_eq (B, C)) ;
        end
    end

    for m = 1:6
        A = sprandn (m, 1, 0.5) ;
        G = GrB (A) ;
        for k = -6:6
            B = diag (A, k) ;
            C = diag (G, k) ;
            assert (gbtest_eq (B, C)) ;
            B = tril (A, k) ;
            C = tril (G, k) ;
            assert (gbtest_eq (B, C)) ;
            B = triu (A, k) ;
            C = triu (G, k) ;
            assert (gbtest_eq (B, C)) ;
        end

        B = diag (A) ;
        C = diag (G) ;
        assert (gbtest_eq (B, C)) ;
        B = tril (A) ;
        C = tril (G) ;
        assert (gbtest_eq (B, C)) ;
        B = triu (A) ;
        C = triu (G) ;
        assert (gbtest_eq (B, C)) ;
    end
end

n = uint64 (2^60) ;
A = magic (5) ;
I = [1 2 3 4 5] ;
H = GrB (n,n) ;
H (I,I) = A ;
d = diag (H) ;
[~,~,x] = find (d) ;
e = diag (A) ;
assert (isequal (e, x))

for k = 1:length(I)
    i = I (k) - 1 ;
    d = diag (H, i) ;
    [~,~,x] = find (d) ;
    e = diag (A, k-1) ;
    assert (isequal (e, x)) ;
    d = diag (H, -i) ;
    [~,~,x] = find (d) ;
    e = diag (A, -(k-1)) ;
    assert (isequal (e, x)) ;
end

I = [1 2 3 n-1 n] ;
H = GrB (n,n) ;
H (I,I) = A ;
d = diag (H, n-2) ;
[~,~,x] = find (d) ;
e = diag (A, 3) ;
assert (isequal (e, x)) ;

fprintf ('\ngbtest25: all tests passed\n') ;

