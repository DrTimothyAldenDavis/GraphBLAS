function gbtest36
%GBTEST36 test abs, sign

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

rng ('default') ;

types = gbtest_types ;
for k = 1:length (types)
    type = types {k} ;

    A = floor (100 * (rand (3, 3) - 0.5)) ;
    A (1,1) = 0 ;

    if (type (1) == 'u')
        A = max (A, 0) ;
    end
    G = gb (A, type) ;
    B = cast (A, type) ;
    assert (isequal (double (B), double (G)))

    H = abs (G) ;
    C = abs (B) ;
    assert (isequal (double (C), double (H)))

    H = sign (G) ;
    if (isequal (type, 'logical'))
        % sign (B) is not defined for MATLAB logical matrices
        C = B ;
    else
        C = sign (B) ;
    end
    assert (isequal (double (C), double (H)))

end

fprintf ('gbtest36: all tests passed\n') ;

