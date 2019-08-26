function gbtest25
%GBTEST25 test diag, tril, triu

for trials = 1:40
    fprintf ('.') ;

    for m = 2:6
        for n = 2:6
            A = sprand (m, n, 0.5) ;
            G = gb (A) ;
            for k = -m:n
                B = diag (A, k) ;
                C = diag (G, k) ;
                assert (isequal (B, double (C))) ;
                B = tril (A, k) ;
                C = tril (A, k) ;
                assert (isequal (B, double (C))) ;
                B = triu (A, k) ;
                C = triu (A, k) ;
                assert (isequal (B, double (C))) ;
            end
            B = diag (A) ;
            C = diag (G) ;
            assert (isequal (B, double (C))) ;
        end
    end

    for m = 1:6
        A = sprandn (m, 1, 0.5) ;
        G = gb (A) ;
        for k = -6:6
            B = diag (A, k) ;
            C = diag (G, k) ;
            assert (isequal (B, double (C))) ;
            B = tril (A, k) ;
            C = tril (G, k) ;
            assert (isequal (B, double (C))) ;
            B = triu (A, k) ;
            C = triu (G, k) ;
            assert (isequal (B, double (C))) ;
        end

        B = diag (A) ;
        C = diag (G) ;
        assert (isequal (B, double (C))) ;
        B = tril (A) ;
        C = tril (G) ;
        assert (isequal (B, double (C))) ;
        B = triu (A) ;
        C = triu (G) ;
        assert (isequal (B, double (C))) ;
    end
end

fprintf ('\ngbtest25: all tests passed\n') ;

