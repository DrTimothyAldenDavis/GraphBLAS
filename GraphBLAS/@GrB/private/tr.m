
clear all
diary tr4.txt
a_size = 28 ;
m_size = 28 ;
n_size = 28 ;
results = nan (4, a_size, m_size, n_size) ;
nz_result = nan (a_size, m_size, n_size) ;
nvec_result = nan (a_size, m_size, n_size) ;
desc.format = 'sparse' ;

for a = 14:a_size
    anz = 2^a ;
    for mk = 2:m_size
        m = 2^mk ;
        fprintf ('\n') ;
        for nk = 2:n_size
            n = 2^nk ;

            d = anz / (m*n) ;
            if (d >= 1)  
                continue ;
            end
            A = GrB.random (m, n, d) ;
            A = GrB (A, 'sparse/hyper') ;
            [f,s] = GrB.format (A) ;

            nz = nnz (A) ;
            nvec = GrB.nonz (A, 'col') ;

            nz_result (a, mk, nk) = nz ;
            nvec_result (a, mk, nk) = nvec ;

            sort_work = log2 (nz + 1) * nz  ;
            bucket_work = (nz + m + nvec) ;

            for method = 1:4

                if (method == 1)
                    % use qsort
                    GB_mex_hack (1) ;
                elseif (method == 2)
                    % use bucket: non-atomic
                    GB_mex_hack (-2) ;
                elseif (method == 3)
                    % use bucket: atomic
                    GB_mex_hack (-3) ;
                else
                    % auto
                    GB_mex_hack (0) ;
                end

                ntrials = 0 ;
                tstart = tic ;
                while (1)
                    C = A' ;
                    ntrials = ntrials + 1 ;
                    t = toc (tstart) ;
                    if (t > 0.1)
                        break ;
                    end
                    clear C
                end
                t = t / ntrials ;

                results (method, a, mk, nk) = t ;

            end

            t1 = results (1, a, mk, nk) ;
            t2 = results (2, a, mk, nk) ;
            t3 = results (3, a, mk, nk) ;
            t4 = results (4, a, mk, nk) ;
            tbucket = min (t2, t3) ;
            if (t2 < t3 * 0.9)
                % non-atomic is faster than atomic
                twhich = 'n' ;
            elseif (t2 > t3 * (1/0.9))
                % atomic is faster than non-atomic
                twhich = 'a' ;
            else
                % atomic and non-atomic about the same
                twhich = ' ' ;
            end

            fprintf ('a %2d m %2d n %2d nv %8d: %s ', a, mk, nk, nvec, s (1)) ;
            fprintf ('%12.6f %12.6f %12.6f (%s) [%12.f] ', ...
                t1, t2, t3, twhich, t4) ;
            fprintf ('rel:[%8.4f %8.4f %8.4f] ', ...
                sort_work/bucket_work, t1/tbucket, ...
                (sort_work/bucket_work) / (t1/tbucket))) ;
            rel = t4 / min ([t1 t2 t3 t4]) ;
            if (rel == 1)
                fprintf ('auto best\n') ;
            elseif (rel < 1.1)
                fprintf ('auto ~\n') ;
            else
                fprintf ('auto %8.4f\n', rel) ;
            end
        end
    end
end

save tr_results results nz_result nvec_result
