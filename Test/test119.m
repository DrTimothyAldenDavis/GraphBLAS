function test119
%TEST119 performance tests for GrB_assign

fprintf ('test119:-------------------  C(I,J) += scalar:\n') ;

rng ('default') ;
n = 4000 ; ;

k = 3000 ;
I.begin = 0 ;
I.inc = 1 ;
I.end = k-1 ;

ncores = feature ('numcores') ;

for dc = [0 1e-6 1e-5 1e-4 1e-3 1e-2 0.1 1]

    C0 = sprand (n, n, dc) ;

    % warmup
    C1 = C0 ;
    C1 (1:k,1:k) = C1 (1:k,1:k) + pi ;

    fprintf ('\n--------------------------------------\n') ;
    fprintf ('dc = %g  nnz(C) %8.4f  million\n', dc, nnz(C0)/1e6) ;
    tic
    C1 = C0 ;
    C1 (1:k,1:k) = C1 (1:k,1:k) + pi ;
    tm = toc ;

    scalar = sparse (pi) ;

    for nthreads = [1 2 4 8 16 20 32 40 64]
        if (nthreads > 2*ncores)
            break ;
        end
        if (nthreads > 1 && t1 < 0.01)
            break ;
        end

        nthreads_set (nthreads) ;

        % default
        GB_mex_hack (0) ;
        C2 = GB_mex_assign (C0, [ ], 'plus', scalar, I, I) ;
        C2 = GB_mex_assign (C0, [ ], 'plus', scalar, I, I) ;
        tg = gbresults ;
        assert (isequal (C1, C2.matrix)) ;
        if (nthreads == 1)
            t1 = tg ;
        end

        % method3
        GB_mex_hack (-1) ;
        C2 = GB_mex_assign (C0, [ ], 'plus', scalar, I, I) ;
        C2 = GB_mex_assign (C0, [ ], 'plus', scalar, I, I) ;
        t_3 = gbresults ;
        assert (isequal (C1, C2.matrix)) ;

        % method8 (always fastest)
        GB_mex_hack (1) ;
        C2 = GB_mex_assign (C0, [ ], 'plus', scalar, I, I) ;
        C2 = GB_mex_assign (C0, [ ], 'plus', scalar, I, I) ;
        t_8 = gbresults ;
        assert (isequal (C1, C2.matrix)) ;

        fprintf ('%3d : MATLAB: %10.4f GB: %8.4f %8.4f %8.4f [%8.2f]', ...
            nthreads, tm, tg, t_3, t_8, t_3 / t_8) ;

        fprintf (' speedup %10.4f %10.4f\n', tm / tg, t1/tg) ;

    end
end

