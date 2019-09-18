function gbtest48
%GBTEST48 test gb.apply

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

rng ('default') ;
desc.kind = 'sparse' ;

for trial = 1:40

    A = rand (4) ;
    A (A > .5) = 0 ;
    G = gb (A) ;

    C0 = -A ;
    C1 = gb.apply ('negate', A) ;
    C2 = gb.apply ('negate', A, desc) ;
    C3 = gb.apply ('negate', G, desc) ;
    C4 = gb.apply ('negate', G) ;

    assert (isequal (C0, C1)) ;
    assert (isequal (C0, C2)) ;
    assert (isequal (C0, C3)) ;
    assert (isequal (C0, C4)) ;

    assert (isequal (class (C2), 'double')) ;
    assert (isequal (class (C3), 'double')) ;

end

fprintf ('gbtest48: all tests passed\n') ;

