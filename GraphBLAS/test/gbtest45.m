function gbtest45
%GBTEST45 test reduce to vector

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

rng ('default') ;
d.kind = 'sparse' ;

for trial = 1:40

    A = rand (4) ;
    G = gb (A) ;
    x = gb.vreduce ('+', A) ;
    y = gb.vreduce ('+', G) ;
    t = gb.vreduce ('+', G, d) ;
    z = sum (G, 2) ;
    w = sum (A, 2) ;
    
    assert (isequal (w, x)) ;
    assert (isequal (w, y)) ;
    assert (isequal (w, z)) ;
    assert (isequal (w, t)) ;

    assert (isequal (class (t), 'double')) ;

end

fprintf ('gbtest45: all tests passed\n') ;

