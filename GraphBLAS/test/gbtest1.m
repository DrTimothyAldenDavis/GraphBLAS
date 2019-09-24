function gbtest1
%GBTEST1 test gb

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

rng ('default') ;
X = 100 * sprand (3, 4, 0.4)

% types = { 'double' } ;

types = gbtest_types ;

m = 2 ;
n = 3 ;

for k = 1:length (types)
    type = types {k} ;

    fprintf ('\n---- A = gb (X) :\n') ;
    A = gb (X)
    Z = double (A)
    assert (gbtest_eq (Z, X)) ;

    fprintf ('\n---- A = gb (X, ''%s'') :\n', type) ;
    A = gb (X, type)
    Z = logical (A)
    if (isequal (type, 'logical'))
        assert (islogical (Z)) ;
        assert (gbtest_eq (Z, logical (X))) ;
    end

    fprintf ('\n---- A = gb (%d, %d) :\n', m, n) ;
    A = gb (m, n)
    Z = sparse (m, n)
    assert (isequal (A, Z)) ;
    A = gb (m, n, 'by row') ;
    assert (isequal (A, Z)) ;

    fprintf ('\n---- A = gb (%d, %d, ''%s'') :\n', m, n, type) ;
    A = gb (m, n, type)
    Z = logical (A)
    if (isequal (type, 'logical'))
        assert (islogical (Z)) ;
        assert (gbtest_eq (Z, logical (sparse (m,n)))) ;
    end

    Z = full (fix (X)) ;
    A = gb (Z, 'by row', type) ;
    Y = cast (Z, type) ;
    assert (isequal (A, Y)) ;

end

X = [ ] ;
A = gb (X, 'by row', 'double') ;
assert (isequal (A, X)) ;

A = gb (m, n, 'by row', 'double') ;
X = sparse (m, n) ;
assert (isequal (A, X)) ;

fprintf ('gbtest1: all tests passed\n') ;

