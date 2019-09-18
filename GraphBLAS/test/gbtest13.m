function gbtest13
%GBTEST13 test find and gb.extracttuples

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

list = gbtest_types ;

A = 100 * rand (3) ;
[I, J, X] = find (A) ;
I_0 = int64 (I) - 1 ;
J_0 = int64 (J) - 1 ;
A (1,1) = 0 ;

d.kind = 'zero-based' ;

for k = 1:length(list)
    xtype = list {k} ;
    fprintf ('testing: %s\n', xtype) ;
    C = cast (A, xtype) ;
    G = gb (C) ;

    [I1, J1, X1] = find (G) ;
    % find drops the zeros
    nz = find (C (:) ~= 0) ;
    assert (isequal (C (nz), X1)) ;
    assert (isequal (I (nz), I1)) ;
    assert (isequal (J (nz), J1)) ;

    [I1, J11] = find (G) ;
    nz = find (C (:) ~= 0) ;
    assert (isequal (I (nz), I1)) ;
    assert (isequal (J (nz), J1)) ;

    % gb.extracttuples returns the zeros
    [I0, J0, X0] = gb.extracttuples (G, d) ;
    assert (isequal (C (:), X0)) ;
    assert (isequal (I_0, I0)) ;
    assert (isequal (J_0, J0)) ;

    [I1] = gb.extracttuples (G, d) ;
    assert (isequal (I1, I0)) ;

end

v = rand (1,3) ;
[i1, j1, x1] = find (v) ;
[i2, j2, x2] = find (gb (v)) ;
assert (isequal (x1, x2)) ;
assert (isequal (i1, i2)) ;
assert (isequal (j1, j2)) ;

[i2, j2] = find (gb (v)) ;
assert (isequal (i1, i2)) ;
assert (isequal (j1, j2)) ;

j1 = find (v) ;
j2 = find (gb (v)) ;
assert (isequal (j1, j2)) ;

fprintf ('gbtest13: all tests passed\n') ;

