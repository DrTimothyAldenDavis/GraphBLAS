function test111
%TEST111 performance test for eWiseAdd

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

fprintf ('\ntest111 performance tests : eWiseAdd \n') ;
rng ('default') ;

save = nthreads_get ;

n = 40e6 ;
% n = 1e6 ;
Empty = sparse (n, 1) ;

for d = [0.001 0.01 0.1:.1:1]

    if (d == 1)
        A = sparse (rand (n,1)) ;
        B = sparse (rand (n,1)) ;
    else
        A = sprand (n,1,d) ;
        B = sprand (n,1,d) ;
    end

    fprintf ('\nd = %g\n', d) ;
    fprintf ('nnz (A) = %d nnz (B) = %d\n', nnz (A), nnz (B)) ;

    %--------------------------------------------------------------------------
    % add
    %--------------------------------------------------------------------------

    % warmup
    C1 = A + B ;
    C1 = A + B ;
    tic
    C1 = A + B ;
    toc
    tm = toc ;
    fprintf ('nnz (C) = %d\n', nnz (C1));
    fprintf ('\nvia GB_add:\n') ;
%    try
        for nthreads = 1:4
            nthreads_set (nthreads) ;
            % tic
            C4 = GB_mex_AplusB (A, B, 'plus') ;
            C4 = GB_mex_AplusB (A, B, 'plus') ;
            C4 = GB_mex_AplusB (A, B, 'plus') ;
            % toc
            tg = gbresults ;
            if (nthreads == 1)
                t1 = tg ;
            end
            fprintf ('nthreads %d GraphBLAS time: %g ', nthreads, tg) ;
            fprintf ('speedup %g over MATLAB: %g\n', t1/tg, tm/tg) ;
            assert (spok (C4) == 1) ;
            assert (isequal (C1, C4)) ;
        end
%    catch
%    end

    %--------------------------------------------------------------------------
    % ewise multiply
    %--------------------------------------------------------------------------

    % warmup
    C1 = A .* B ;
    C1 = A .* B ;
    tic
    C1 = A .* B ;
    toc
    tm = toc ;
    fprintf ('nnz (C) = %d for A.*B\n', nnz (C1));
    fprintf ('\nvia GB_eWiseMult:\n') ;
%    try
        for nthreads = 1:4
            nthreads_set (nthreads) ;
            % tic
% usage: w = GB_mex_eWiseMult_Vector (w, mask, accum, mult, u, v, desc)

            C4 = GB_mex_eWiseMult_Vector (Empty, [ ], [ ], 'times', A,B, [ ]) ;
            C4 = GB_mex_eWiseMult_Vector (Empty, [ ], [ ], 'times', A,B, [ ]) ;
            C4 = GB_mex_eWiseMult_Vector (Empty, [ ], [ ], 'times', A,B, [ ]) ;
%           C4 = GB_mex_AplusB (A, B, 'plus') ;
%           C4 = GB_mex_AplusB (A, B, 'plus') ;
%           C4 = GB_mex_AplusB (A, B, 'plus') ;
            % toc
            tg = gbresults ;
            if (nthreads == 1)
                t1 = tg ;
            end
            fprintf ('nthreads %d GraphBLAS time: %g ', nthreads, tg) ;
            fprintf ('speedup %g over MATLAB: %g\n', t1/tg, tm/tg) ;
            assert (spok (C4.matrix) == 1) ;
            assert (isequal (C1, C4.matrix)) ;
        end
%    catch
%    end

end

fprintf ('\ndense matrices:\n') ;

A = full (A) ;
B = full (B) ;

for trial = 1:4
    tic
    C1 = A + B ;
    toc
    tm = toc ;
end
nthreads_set (save) ;

fprintf ('test111: all tests passed\n') ;
