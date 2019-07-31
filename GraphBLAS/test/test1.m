function test
%TEST1 test gb

rng ('default') ;
X = 100 * sprand (3, 4, 0.4)

% types = { 'double' } ;

types = list_types ;

m = 2 ;
n = 3 ;

for k = 1:length (types)
    type = types {k} ;

    fprintf ('\n---- A = gb:\n') ;
    A = gb
    Z = sparse (A)

    fprintf ('\n---- A = gb (X) :\n') ;
    A = gb (X)
    Z = sparse (A)
    assert (spok (Z) == 1) ;
    assert (isequal (Z, X)) ;

    fprintf ('\n---- A = gb (''%s'') :\n', type) ;
    A = gb (type)
    Z = sparse (A)

    fprintf ('\n---- A = gb (X, ''%s'') :\n', type) ;
    A = gb (X, type)
    Z = sparse (A)
    if (isequal (type, 'logical'))
        assert (islogical (Z)) ;
        assert (isequal (Z, logical (X))) ;
    end

    fprintf ('\n---- A = gb (%d, %d) :\n', m, n) ;
    A = gb (m, n)
    Z = sparse (A)

    fprintf ('\n---- A = gb (%d, %d, ''%s'') :\n', m, n, type) ;
    A = gb (m, n, type)
    Z = sparse (A)
    if (isequal (type, 'logical'))
        assert (islogical (Z)) ;
        assert (isequal (Z, logical (sparse (m,n)))) ;
    end

end

