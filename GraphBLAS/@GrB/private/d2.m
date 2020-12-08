
% dot2 rule

clear all
diary d2_Dec8_slash.txt
desc.in0 = 'transpose' ;
GrB (1) ;

kset = [1 5 10:10:100 200:300:1000] ;
nset = 2.^(4:2:32) ;
aset = 2.^(16:4:24) ;

Anz   = nan (length (nset), length (kset), length (aset)) ;
Bnz   = nan (length (nset), length (kset), length (aset)) ;
Anvec = nan (length (nset), length (kset), length (aset)) ;
Bnvec = nan (length (nset), length (kset), length (aset)) ;
Csize = nan (length (nset), length (kset), length (aset)) ;
Tdot  = nan (length (nset), length (kset), length (aset)) ;
Tsax  = nan (length (nset), length (kset), length (aset)) ;
Tauto = nan (length (nset), length (kset), length (aset)) ;
K     = nan (length (nset), length (kset), length (aset)) ;

for kk = 1:length (kset)
    k = kset (kk) ;
    fprintf ('\nk: %d\n', k) ;

    for nk = 1:length (nset)
        n = nset (nk) ;

        for ak = 1:length (aset)
            anz = aset (ak) ;

            d = anz / (n*k) ;
            if (d > 1)
                continue ;
            end

            A = GrB.random (n, k, d) ;
            B = GrB.random (n, k, d) ;
            Anz (nk, kk, ak) = nnz (A) ;
            Bnz (nk, kk, ak) = nnz (B) ;
            Csize (nk, kk, ak) = k^2 ;
            K (nk, kk, ak) = k ;
            N (nk, kk, ak) = n ;
            anvec = GrB.entries (A, 'col') ;
            Anvec (nk, kk, ak) = anvec ;
            Bnvec (nk, kk, ak) = GrB.entries (B, 'col') ;

            % force dot
            GB_mex_hack (10) ;
            % warmup
            C = GrB.mxm (A, '+.*', B, desc) ;
            ntrials = 0 ;
            tstart = tic ;
            while (1)
                C = GrB.mxm (A, '+.*', B, desc) ;
                ntrials = ntrials + 1 ;
                tdot = toc (tstart) ;
                if (tdot > 0.1)
                    break ;
                end
                clear C
            end
            tdot = tdot / ntrials ;

            % force saxpy
            GB_mex_hack (11) ;
            % warmup
            C = GrB.mxm (A, '+.*', B, desc) ;
            ntrials = 0 ;
            tstart = tic ;
            while (1)
                C = GrB.mxm (A, '+.*', B, desc) ;
                ntrials = ntrials + 1 ;
                tsax = toc (tstart) ;
                if (tsax > 0.1)
                    break ;
                end
                clear C
            end
            tsax = tsax / ntrials ;

            % auto
            GB_mex_hack (0) ;
            % warmup
            GrB.burble (1) ;
            C = GrB.mxm (A, '+.*', B, desc) ;
            GrB.burble (0) ;
            ntrials = 0 ;
            tstart = tic ;
            while (1)
                C = GrB.mxm (A, '+.*', B, desc) ;
                ntrials = ntrials + 1 ;
                tauto = toc (tstart) ;
                if (tauto > 0.1)
                    break ;
                end
                clear C
            end
            tauto = tauto / ntrials ;

            fprintf ('anz %10d bnz %10d n %10d k %4d nvec %4d ', ...
                nnz (A), nnz (B), n, k, anvec) ;

            fprintf ('tdot %12.6f tsax %12.6f tdot/tsax %12.6f', ...
                tdot, tsax, tdot/tsax) ;

            fprintf (' tauto: %12.6f ', tauto) ;

            rel = tauto / min ([tauto tsax tdot]) ;
            if (rel == 1)
                fprintf ('auto best\n') ;
            elseif (rel < 1.1)
                fprintf ('auto ~\n') ;
            else
                fprintf ('auto %8.4f\n', rel) ;
            end

            Tdot (nk, kk, ak) = tdot ;
            Tsax (nk, kk, ak) = tsax ;
            Tauto (nk, kk, ak) = tauto ;
        end
    end
end

