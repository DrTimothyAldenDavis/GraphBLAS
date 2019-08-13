function test21
%TEST21 test isfinite, isinf, isnan

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

rng ('default') ;
for trial = 1:40
    fprintf ('.') ;
    for m = 0:10
        for n = 0:10
            A = 100 * sprand (m, n, 0.5) ;
            A (1,1) = nan ;
            A (2,2) = inf ;
            G = gb (A) ;

            assert (isequal (isfinite (A), full (isfinite (G)))) ;
            assert (isequal (isinf    (A), sparse (isinf    (G)))) ;
            assert (isequal (isnan    (A), sparse (isnan    (G)))) ;
        end
    end
end

fprintf ('\ntest21: all tests passed\n') ;

