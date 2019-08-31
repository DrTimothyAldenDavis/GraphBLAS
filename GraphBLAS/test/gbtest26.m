function gbtest26
%GBTEST26 test typecasting

types = gbtest_types ;

rng ('default') ;
for k1 = 1:length (types)

    atype = types {k1} ;
    fprintf ('\n================================================ %s\n', atype) ;
    A = cast (100 * rand (3), atype)
    H = gb (A) ;
    B = cast (H, atype) ;
    assert (gbtest_eq (A, B)) ;

    for k2 = 1:length (types)

        gtype = types {k2} ;
        fprintf ('\n------------ %s:\n', gtype) ;
        G = gb (H, gtype)
        K = gb (G, atype)
        C = cast (G, atype)
    end
end

fprintf ('gbtest26: all tests passed\n') ;

