% GBTEST

% gbmake

rng ('default') ;
X = 100 * sprand (3, 4, 0.4)

% types = { 'double' } ;

types = {
    'int8'
    'int16'
    'int32'
    'int64'
    'uint8'
    'uint16'
    'uint32'
    'uint64'
    'single'
    'double'
    'logical' } ;
%    'complex' } ;

m = 2 ;
n = 3 ;

for k = 1:length (types)
    type = types {k} ;

    fprintf ('\n---- A = gbnew:\n') ;
    A = gbnew
    Z = gbsparse (A)

    fprintf ('\n---- A = gbnew (X) :\n') ;
    A = gbnew (X)
    Z = gbsparse (A)
    assert (spok (Z) == 1) ;
    assert (isequal (Z, X)) ;

    fprintf ('\n---- A = gbnew (''%s'') :\n', type) ;
    A = gbnew (type)
    Z = gbsparse (A)

    fprintf ('\n---- A = gbnew (X, ''%s'') :\n', type) ;
    A = gbnew (X, type)
    Z = gbsparse (A)
    if (isequal (type, 'logical'))
        assert (islogical (Z)) ;
        assert (isequal (Z, logical (X))) ;
    end

    fprintf ('\n---- A = gbnew (%d, %d) :\n', m, n) ;
    A = gbnew (m, n)
    Z = gbsparse (A)

    fprintf ('\n---- A = gbnew (%d, %d, ''%s'') :\n', m, n, type) ;
    A = gbnew (m, n, type)
    Z = gbsparse (A)
    if (isequal (type, 'logical'))
        assert (islogical (Z)) ;
        assert (isequal (Z, logical (sparse (m,n)))) ;
    end

end


