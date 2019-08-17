function gbtest22
%GBTEST22 test reduce to scalar

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

rng ('default') ;
for trial = 1:40
    fprintf ('.') ;
    for m = 0:10
        for n = 0:10
            A = 100 * sprand (m, n, 0.5) ;
            G = gb (A) ;
            [i j x] = find (A) ;

            % c1 = sum (A, 'all') ;
            c1 = sum (sum (A)) ;
            c2 = gb.reduce ('+', A) ;
            c3 = sum (G, 'all') ;
            assert (logical (norm (c1-c2,1) <= 1e-12 * norm (c1,1))) ;
            assert (logical (norm (c1-c3,1) <= 1e-12 * norm (c1,1))) ;

            % c1 = pi + sum (A, 'all') ;
            c1 = pi + sum (sum (A)) ;
            c2 = gb.reduce (pi, '+', '+', A) ;
            c3 = pi + sum (G, 'all') ;
            assert (logical (norm (c1-c2,1) <= 1e-12 * norm (c1,1))) ;
            assert (logical (norm (c1-c3,1) <= 1e-12 * norm (c1,1))) ;

            % c1 = prod (x, 'all') ;
            c1 = prod (x) ;
            c2 = gb.reduce ('*', A) ;
            assert (logical (norm (c1-c2,1) <= 1e-12 * norm (c1,1))) ;

            % c1 = prod (A, 'all') ;
            c1 = prod (prod (A)) ;
            c2 = prod (G, 'all') ;
            assert (logical (norm (c1-c2,1) <= 1e-12 * norm (c1,1))) ;

            % c1 = pi + prod (x, 'all') ;
            c1 = pi + prod (x) ;
            c2 = gb.reduce (pi, '+', '*', A) ;
            assert (logical (norm (c1-c2,1) <= 1e-12 * norm (c1,1))) ;

            % c1 = max (A, [ ], 'all') ;
            c1 = max (max (A)) ;
            c2 = gb.reduce ('max', A) ;
            if (nnz (A) < m*n)
                c2 = max (full (c2), 0) ;
            end
            c3 = max (G, [ ], 'all') ;
            assert (logical (norm (c1-c2,1) <= 1e-12 * norm (c1,1))) ;
            assert (logical (norm (c1-c3,1) <= 1e-12 * norm (c1,1))) ;

            % c1 = min (A, [ ], 'all') ;
            c1 = min (min (A)) ;
            c2 = gb.reduce ('min', A) ;
            if (nnz (A) < m*n)
                c2 = min (full (c2), 0) ;
            end
            c3 = min (G, [ ], 'all') ;
            assert (logical (norm (c1-c2,1) <= 1e-12 * norm (c1,1))) ;
            assert (logical (norm (c1-c3,1) <= 1e-12 * norm (c1,1))) ;

            B = logical (A) ;
            G = gb (B) ;

            % c1 = any (A, 'all') ;
            c1 = any (any (A)) ;
            c2 = gb.reduce ('|.logical', A) ;
            c3 = any (G, 'all') ;
            assert (logical (c1 == logical (c2))) ;
            assert (logical (c1 == logical (c3))) ;

            % c1 = all (A, 'all') ;
            c1 = all (all (A)) ;
            c3 = all (G, 'all') ;
            assert (logical (c1 == logical (c3))) ;

            [i j x] = find (A) ;
            % c1 = all (x, 'all') ;
            c1 = all (x) ;
            c2 = gb.reduce ('&.logical', A) ;
            assert (logical (c1 == logical (c2))) ;

        end
    end
end

fprintf ('\ngbtest22: all tests passed\n') ;

