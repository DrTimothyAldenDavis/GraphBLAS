function gbtest31
%GBTEST31 test gb and casting

types = gbtest_types ;
fprintf ('gbtest31: typecasting\n') ;
rng ('default') ;

for k = 1:length (types)
    type = types {k} ;
    fprintf ('%s ', type) ;

    for m = 0:5
        for n = 0:5
            A = zeros (m, n, type) ;
            G = gb (m, n, type) ;
            C = cast (G, type) ;
            assert (isequal (A, C)) ;
        end
    end

    A = 100 * rand (5, 5) ;
    A = cast (A, type) ;
    G = gb (A) ;

    G2 = sparse (G) ;
    assert (isequal (G, G2)) ;

    for k2 = 1:length (types)
        type2 = types {k} ;
        G2 = gb (G, type2) ;
        A2 = cast (A, type2) ;
        C = cast (G2, type2) ;
        assert (isequal (A2, C)) ;
    end

    F = 100 * ones (5, 5) ;
    F = cast (F, type) ;
    id = F (1,1) ;

    A = 100 * sparse (diag (1:5)) ;

    G = gb (A, type) ;
    G2 = gb (F) ;
    G2 (logical (speye (5))) = 100:100:500 ;

    for k2 = 1:length (types)
        type2 = types {k} ;
        G3 = full (G, type2, id) ;
        G4 = gb (G2, type2) ;
        assert (isequal (G3, G4)) ;
        assert (isequal (double (G3), double (G4))) ;
        assert (isequal (single (G3), single (G4))) ;
        assert (isequal (uint16 (G3), uint16 (G4))) ;
    end

end

fprintf ('\ngbtest31: all tests passed\n') ;
