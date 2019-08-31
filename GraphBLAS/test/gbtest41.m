function gbtest41
%GBTEST41 test ones, zeros, false

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

types = gbtest_types ;

for trial = 1:40
    fprintf ('.') ;

    for k = 1:length(types)
        type = types {k} ;
        G = gb (rand (2), type) ;

        G2 = ones (3, 4, 'like', G) ;
        G3 = gb (ones (3, 4, type)) ;
        assert (gbtest_eq (G2, G3)) ;

        G2 = zeros (3, 4, 'like', G) ;
        G3 = gb (zeros (3, 4, type)) ;
        assert (isequal (gb.type (G2), gb.type (G3))) ;
        assert (isequal (type, gb.type (G3))) ;
        assert (norm (double (G2) - double (G3), 1) == 0) ;

        if (isequal (type, 'logical'))
            G2 = false (3, 4, 'like', G) ;
            G3 = gb (false (3, 4, type)) ;
            assert (isequal (gb.type (G2), gb.type (G3))) ;
            assert (isequal (type, gb.type (G3))) ;
            assert (norm (double (G2) - double (G3), 1) == 0) ;
        end

    end

end

fprintf ('\ngbtest41: all tests passed\n') ;

