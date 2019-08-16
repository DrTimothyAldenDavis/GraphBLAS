function gbtest28
%GBTEST28 test eye and speye

types = gbtest_types ;

for m = -1:10
    fprintf ('.') ;

    A = eye (m) ;
    G = gb.eye (m) ;
    assert (isequal (A, full (double (G)))) ;

    for n = -1:10

        A = eye (m, n) ;
        G = gb.eye (m, n) ;
        assert (isequal (A, full (double (G)))) ;

        for k = 1:length (types)
            type = types {k} ;

            A = eye (m, n, type) ;
            G = gb.eye (m, n, type) ;
            assert (isequal (A, full (double (G)))) ;

            A = eye ([m n], type) ;
            G = gb.eye ([m n], type) ;
            assert (isequal (A, full (double (G)))) ;

            A = eye (m, type) ;
            G = gb.eye (m, type) ;
            assert (isequal (A, full (double (G)))) ;
        end
    end
end

fprintf ('\ngbtest28: all tests passed\n') ;
