function gbtest46
%GBTEST46 test gb.subassign and gb.assign

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

rng ('default') ;
d.kind = 'sparse' ;

for trial = 1:40

    A = rand (4) ;
    G = gb (A) ;
    pg = gb (pi) ;

    C1 = A ;
    C1 (1:3,1) = pi ;

    C2 = gb.subassign (A, pi, { 1:3}, { 1 }) ;
    C3 = gb.subassign (G, pi, { 1:3}, { 1 }) ;
    C4 = gb.subassign (G, pg, { 1:3}, { 1 }) ;
    C5 = gb.subassign (G, pg, { 1:3}, { 1 }, d) ;
    assert (isequal (C1, C2)) ;
    assert (isequal (C1, C3)) ;
    assert (isequal (C1, C4)) ;
    assert (isequal (C1, C5)) ;
    assert (isequal (class (C5), 'double')) ;

    C2 = gb.assign (A, pi, { 1:3}, { 1 }) ;
    C3 = gb.assign (G, pi, { 1:3}, { 1 }) ;
    C4 = gb.assign (G, pg, { 1:3}, { 1 }) ;
    C5 = gb.assign (G, pg, { 1:3}, { 1 }, d) ;
    assert (isequal (C1, C2)) ;
    assert (isequal (C1, C3)) ;
    assert (isequal (C1, C4)) ;
    assert (isequal (C1, C5)) ;
    assert (isequal (class (C5), 'double')) ;

end

fprintf ('gbtest46: all tests passed\n') ;

