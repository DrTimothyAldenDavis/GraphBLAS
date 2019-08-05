function test13
%TEST13 test find and gb.extracttuples
% clear all

list = list_types ;

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
    assert (isequal (C (:), X1)) ;
    assert (isequal (I, I1)) ;
    assert (isequal (J, J1)) ;

    [I0, J0, X0] = gb.extracttuples (G, d) ;
    assert (isequal (C (:), X0)) ;
    assert (isequal (I_0, I0)) ;
    assert (isequal (J_0, J0)) ;
end

fprintf ('test13: all tests passed\n') ;

