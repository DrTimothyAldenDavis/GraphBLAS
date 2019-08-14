function gbtest23
%TEST23 test min and max

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

            c1 = max (MA) ;
            c2 = max (GA) ;
            assert (isequal (sparse (c1), sparse (c2))) ;

            c1 = min (MA) ;
            c2 = min (GA) ;
            assert (isequal (sparse (c1), sparse (c2))) ;

            C1 = max (MA,MB) ;
            C2 = max (MA,GB) ;
            C3 = max (GA,MB) ;
            C4 = max (GA,GB) ;
            assert (isequal (C1, sparse (C2))) ;
            assert (isequal (C1, sparse (C3))) ;
            assert (isequal (C1, sparse (C4))) ;

            C1 = min (MA,MB) ;
            C2 = min (MA,GB) ;
            C3 = min (GA,MB) ;
            C4 = min (GA,GB) ;
            assert (isequal (C1, sparse (C2))) ;
            assert (isequal (C1, sparse (C3))) ;
            assert (isequal (C1, sparse (C4))) ;

            c1 = max (MA, [ ], 'all') ;
            c2 = max (GA, [ ], 'all') ;
            assert (isequal (sparse (c1), sparse (c2))) ;

            c1 = min (MA, [ ], 'all') ;
            c2 = min (GA, [ ], 'all') ;
            assert (isequal (sparse (c1), sparse (c2))) ;

            C1 = max (MA, [ ], 1) ;
            C2 = max (GA, [ ], 1) ;
            assert (isequal (C1, sparse (C2))) ;

            C1 = min (MA, [ ], 1) ;
            C2 = min (GA, [ ], 1) ;
            assert (isequal (C1, sparse (C2))) ;

            C1 = max (MA, [ ], 2) ;
            C2 = max (GA, [ ], 2) ;
            assert (isequal (C1, sparse (C2))) ;

            C1 = min (MA, [ ], 2) ;
            C2 = min (GA, [ ], 2) ;
            assert (isequal (C1, sparse (C2))) ;

        end
    end
end

fprintf ('\ngbtest23: all tests passed\n') ;

