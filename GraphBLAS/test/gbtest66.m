function gbtest66
%GBTEST66 test gb.incidence

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

rng ('default') ;

ok = true ;
try
    E = gb.incidence (ones (4,3)) ;
    ok = false ;
catch expected_error
    expected_error
end
assert (ok) ;

ok = true ;
try
    E = gb.incidence (ones (4), 'gunk') ;
    ok = false ;
catch expected_error
    expected_error
end
assert (ok) ;

for trial = 1:2

    if (trial == 1)
        ij = [
        4 1
        1 2
        4 3
        6 3
        7 3
        1 4
        7 4
        2 5
        7 5
        3 6
        5 6
        2 7 ] ;
        W = sparse (ij (:,1), ij (:,2), ones (12,1), 8, 8) ;
    else
        load west0479 ;
        W = west0479 ;
    end

    W = spones (gb.offdiag (W)) ;
    A = digraph (W) ;
    G = gb (W) ;

    E0 = incidence (A) ;
    E1 = gb.incidence (G) ;
    % E0 and E1 are the same, except the columns are in different orders
    E0 = sortrows (E0')' ;
    E1 = double (E1) ;
    E1 = sortrows (E1')' ;
    assert (isequal (E0, E1)) ;

    E1 = gb.incidence (G, 'int8') ;
    assert (isequal (gb.type (E1), 'int8')) ;
    E1 = double (E1) ;
    E1 = sortrows (E1')' ;
    assert (isequal (E0, E1)) ;

    W = W+W' ;
    A = graph (W) ;
    G = gb (W) ;

    E0 = incidence (A) ;
    E1 = gb.incidence (G, 'upper') ;
    E0 = sortrows (E0')' ;
    E1 = double (E1) ;
    E1 = sortrows (E1')' ;
    assert (isequal (E0, E1)) ;

    E1 = gb.incidence (G, 'lower') ;
    E1 = -E1 ;
    E1 = double (E1) ;
    E1 = sortrows (E1')' ;
    assert (isequal (E0, E1)) ;

end

fprintf ('gbtest66: all tests passed\n') ;

