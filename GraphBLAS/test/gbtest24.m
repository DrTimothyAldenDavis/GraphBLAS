function gbtest24
%GBTEST24 test any, all

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

rng ('default') ;
for trial = 1:40
    for m = 1:10
        if (mod (m,2) == 0)
            fprintf ('.') ;
        end
        for n = 1:10

            MA = sprand (m, n, 0.5) ;
            S = -(sprand (m, n, 0.5) > 0.5) ;
            MA = MA .* S ;

            MB = sprand (m, n, 0.5) ;
            S = -(sprand (m, n, 0.5) > 0.5) ;
            MB = MB .* S ;

            GA = gb (MA) ;
            GB = gb (MB) ;

            c1 = all (MA) ;
            c2 = all (GA) ;
            assert (isequal (c1, double (c2))) ;

            c1 = any (MA) ;
            c2 = any (GA) ;
            assert (isequal (c1, double (c2))) ;

            c1 = all (MA, 'all') ;
            c2 = all (GA, 'all') ;
            assert (isequal (c1, double (c2))) ;

            c1 = any (MA, 'all') ;
            c2 = any (GA, 'all') ;
            assert (isequal (c1, double (c2))) ;

            C1 = all (MA, 1) ;
            C2 = all (GA, 1) ;
            assert (isequal (C1, double (C2))) ;

            C1 = any (MA, 1) ;
            C2 = any (GA, 1) ;
            assert (isequal (C1, double (C2))) ;

            C1 = all (MA, 2) ;
            C2 = all (GA, 2) ;
            assert (isequal (C1, double (C2))) ;

            C1 = any (MA, 2) ;
            C2 = any (GA, 2) ;
            assert (isequal (C1, double (C2))) ;

        end
    end
end

fprintf ('\ngbtest24: all tests passed\n') ;

