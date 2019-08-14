function gbtest25
%TEST25 test diag

for trials = 1:40
    fprintf ('.') ;

    for m = 2:10
        for n = 2:10
            A = sprand (m, n, 0.5) ;
            G = gb (A) ;
            for k = -m:n
                B = diag (A, k) ;
                C = diag (G, k) ;
                assert (isequal (B, sparse (C))) ;
            end
            B = diag (A) ;
            C = diag (G) ;
            assert (isequal (B, sparse (C))) ;
        end
    end

    for m = 1:10
        A = sprandn (m, 1, 0.5) ;
        G = gb (A) ;
        for k = -10:10
            B = diag (A, k) ;
            C = diag (G, k) ;
            assert (isequal (B, sparse (C))) ;
        end
        B = diag (A) ;
        C = diag (G) ;
        assert (isequal (B, sparse (C))) ;
    end
end

fprintf ('\ngbtest25: all tests passed\n') ;

