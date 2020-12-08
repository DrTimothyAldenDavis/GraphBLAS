
% dot2 rule, GraphBLAS:auto vs MATLAB

clear all

% ensure the hack control in GB_AxB_dot2_control is disabled, 
% to use the auto method selection
GB_mex_hack (0) ;

diary d3_Dec8a_hyper.txt
!hostname
feature ('numcores') ;
ver
desc.in0 = 'transpose' ;
GrB (1) ;

kset = [1 5 10:10:100 200:300:1000] ;
nset = 2.^(4:2:32) ;
aset = 2.^(16:4:24) ;

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

            fprintf ('\n') ;
            A = GrB.random (n, k, d) ;
            B = GrB.random (n, k, d) ;
            anvec = GrB.entries (A, 'col') ;

            % ensure A and B are sparse, for fairer comparison w/ GrB:
            A = GrB (A, 'sparse') ;
            B = GrB (A, 'sparse') ;

            % auto
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

            % MATLAB
            A = double (A) ;
            B = double (B) ;
            GrB.burble (1) ;
            % warmup
            C = A'*B ;
            ntrials = 0 ;
            tstart = tic ;
            while (1)
                C = A'*B ;
                ntrials = ntrials + 1 ;
                tmatlab = toc (tstart) ;
                if (tmatlab > 0.1)
                    break ;
                end
                clear C
            end
            tmatlab = tmatlab / ntrials ;
            GrB.burble (0) ;

            fprintf ('anz %10d bnz %10d n %10d k %4d nvec %4d ', ...
                nnz (A), nnz (B), n, k, anvec) ;

            fprintf ('tmatlab %12.6f tgrb %12.6f tmatlab/tgrb %12.6f', ...
                tmatlab, tauto, tmatlab/tauto) ;

            rel = tmatlab / tauto ;
            if (rel < 1)
                fprintf (' OUCH') ;
            end
            fprintf ('\n') ;

        end
    end
end

