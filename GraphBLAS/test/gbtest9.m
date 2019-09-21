function gbtest9
%GBTEST9 test eye and speye

types = gbtest_types ;

A = eye ;
G = gb.eye ;
assert (gbtest_eq (A, G)) ;
G = gb.speye ;
assert (gbtest_eq (A, G)) ;

for m = -1:10
    fprintf ('.') ;

    A = eye (m) ;
    G = gb.eye (m) ;
    assert (gbtest_eq (A, G)) ;
    G = gb.speye (m) ;
    assert (gbtest_eq (A, G)) ;

    for n = -1:10

        A = eye (m, n) ;
        G = gb.eye (m, n) ;
        assert (gbtest_eq (A, G)) ;
        G = gb.speye (m, n) ;
        assert (gbtest_eq (A, G)) ;

        for k = 1:length (types)
            type = types {k} ;

            A = eye (m, n, type) ;
            G = gb.eye (m, n, type) ;
            assert (gbtest_eq (A, G)) ;
            G = gb.speye (m, n, type) ;
            assert (gbtest_eq (A, G)) ;

            A = eye ([m n], type) ;
            G = gb.eye ([m n], type) ;
            assert (gbtest_eq (A, G)) ;
            G = gb.speye ([m n], type) ;
            assert (gbtest_eq (A, G)) ;

            A = eye (m, type) ;
            G = gb.eye (m, type) ;
            assert (gbtest_eq (A, G)) ;
            G = gb.speye (m, type) ;
            assert (gbtest_eq (A, G)) ;

        end
    end
end

fprintf ('\ngbtest9: all tests passed\n') ;

