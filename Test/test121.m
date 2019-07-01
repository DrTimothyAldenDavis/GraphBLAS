function test121
%TEST121 performance tests for GrB_assign

fprintf ('test121:---------------- C(I,J)+=A performance\n') ;

rng ('default') ;
n = 1e6 ;
k = n/10 ;

%   I.begin = 0 ;
%   I.inc = 1 ;
%   I.end = k-1 ;
    I1 = randperm (k) ;
    I0 = uint64 (I1) - 1 ;

ncores = feature ('numcores') ;

for dc = [ 0 1e-6 1e-5 1e-4 ]

    for da = [ 0 1e-6 1e-5 1e-4 1e-3 ]

        % warmup
        C1 = C0 ;
        C1 (1:k,1:k) = C1 (1:k,1:k) + A ;

        fprintf ('\n--------------------------------------\n') ;
        fprintf ('dc = %g, da = %g\n', dc, da) ;
        tic
        C1 = C0 ;
        % C1 (1:k,1:k) = C1 (1:k,1:k) + A ;
        C1 (I1,I1) = C1 (I1,I1) + A ;
        tm = toc ;

        for nthreads = [1 2 4 8 16 20 32 40 64]
            if (nthreads > 2*ncores)
                break ;
            end
            nthreads_set (nthreads) ;
            if (nthreads > 1 && t1 < 0.01)
                continue ;
            end

            % default
            GB_mex_hack (0) ;
            C2 = GB_mex_assign (C0, [ ], 'plus', A, I0, I0) ;
            C2 = GB_mex_assign (C0, [ ], 'plus', A, I0, I0) ;
            tg = gbresults ;
            assert (isequal (C1, C2.matrix)) ;
            if (nthreads == 1)
                t1 = tg ;
            end

            % method5, sometimes 10% faster for very very sparse case
            GB_mex_hack (-1) ;
            C2 = GB_mex_assign (C0, [ ], 'plus', A, I0, I0) ;
            C2 = GB_mex_assign (C0, [ ], 'plus', A, I0, I0) ;
            t_5 = gbresults ;
            assert (isequal (C1, C2.matrix)) ;
            if (nthreads == 1)
                t1 = t_5 ;
            end

            % method10 (typically tied, fastest, or almost so)
            GB_mex_hack (1) ;
            C2 = GB_mex_assign (C0, [ ], 'plus', A, I0, I0) ;
            C2 = GB_mex_assign (C0, [ ], 'plus', A, I0, I0) ;
            t_10 = gbresults ;
            assert (isequal (C1, C2.matrix)) ;
            if (nthreads == 1)
                t1 = t_10 ;
            end

            GB_mex_hack (0) ;

            fprintf ('%3d : MATLAB: %10.4f GB: %10.4f %10.4f %10.4f', ...
                nthreads, tm, tg, t_5, t_10) ;

            fprintf (' [ %10.4f ] speedup %10.4f\n', t_5/t_10, tm / tg) ;
        end
    end
end

