function test117
%TEST117 performance tests for GrB_assign

% test C(:,:)<M> = A

fprintf ('test117 ----------------------------------- C(:,:)<M> = A\n') ;

rng ('default') ;
n = 4000 ;
S = sparse (n,n) ;

I.begin = 0 ;
I.inc = 1 ;
I.end = n-1 ;

ncores = feature ('numcores') ;

for da = [1e-5 1e-4 1e-3 1e-2 1e-1 0.5]
    A  = sprand (n, n, da) ;

for dm = [1e-5 1e-4 1e-3 1e-2 1e-1 0.5]

    M = spones (sprand (n, n, dm)) ;

        fprintf ('\n--------------------------------------\n') ;
        fprintf ('da: %g, dm: %g ', da, dm) ;
        fprintf ('\n') ;

        fprintf ('nnz(M): %g million, ',  nnz (M) / 1e6) ;
        fprintf ('nnz(A): %g million\n',  nnz (A) / 1e6) ;

        % warmup
        C1 = M.*A ;

        tic
        C1 = M.*A ;
        tm = toc ;

        t1 = 0 ;

        for nthreads = [1 2 4 8 16 20 32 40 64]
            if (nthreads > 2*ncores)
                break ;
            end
            nthreads_set (nthreads) ;

            if (nthreads > 1 & t1 < 0.1)
                continue
            end

            % warmup: default method (13d)
            C2 = GB_mex_assign (S, M, [ ], A, I, I) ;
            C2 = GB_mex_assign (S, M, [ ], A, I, I) ;
            tg = gbresults ;
            assert (isequal (C1, C2.matrix)) ;
            if (nthreads == 1)
                t1 = tg ;
            end

%           % nondefault method (15)
%           GB_mex_hack (-1) ;
%           C2 = GB_mex_assign (S, M, [ ], A, I, I) ;
%           C2 = GB_mex_assign (S, M, [ ], A, I, I) ;
%           GB_mex_hack (0) ;
%           tg2 = gbresults ;
            tg2 = inf ;

            assert (isequal (C1, C2.matrix)) ;

            % ewise
            C2 = GB_mex_eWiseMult_Matrix (S, [ ], [ ], 'times', M, A) ;
            C2 = GB_mex_eWiseMult_Matrix (S, [ ], [ ], 'times', M, A) ;
            tg3 = gbresults ;

            assert (isequal (C1, C2.matrix)) ;

            fprintf (...
                '%3d : %8.4f GB: %8.4f %8.4f %8.4f rel %8.2f %8.2f %8.2f\n', ...
                nthreads, tm, tg, tg2, tg3, ...
                    tm / tg, tm/tg2, tm/tg3) ;

        end

    end
end
end
