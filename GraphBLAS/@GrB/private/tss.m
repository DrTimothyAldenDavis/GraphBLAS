
clear all
diary tss_results2.txt
!hostname
index = ssget ;
GrB (1) ;

% sort by nnz
[ignore, list] = sort (index.nnz) ;
nmat = length (list)

T_matlab = nan (nmat, 1) ;
T_grb = nan (nmat, 4) ;

feature ('numcores')
nthreads = feature ('numcores') * 2
GrB.threads (nthreads) ;

Stats_nz = nan (nmat, 1) ;
Stats_m  = nan (nmat, 1) ;
Stats_n  = nan (nmat, 1) ;
Stats_m2 = nan (nmat, 1) ;
Stats_n2 = nan (nmat, 1) ;

for k = 1:nmat

    % get the problem
    id = list (k) ;
    Prob = ssget (id, index) ;
    A = Prob.A ;
    name = Prob.name ;

    % warmup MATLAB and get statistics
    [m n] = size (A) ;
    nz = nnz (A) ;
    n2 = GrB.entries (A, 'col') ;   % # non-empty columns
    C = A' ;        % warmup
    m2 = GrB.entries (C, 'col') ;   % # non-empty rows of A
    clear C
    fprintf ('%-40s m: %9d ', name, m) ;
    if (m == m2)
        fprintf ('(         ) ') ;
    else
        fprintf ('(%9d) ', m2) ;
    end
    fprintf ('n: %9d ', m) ;
    if (n == n2)
        fprintf ('(         ) ') ;
    else
        fprintf ('(%9d) ', n2) ;
    end
    fprintf ('nz: %11d ', nz) ;

    Stats_nz (k) = nz ;
    Stats_m  (k) = m ;
    Stats_n  (k) = n ;
    Stats_m2 (k) = m2 ;
    Stats_n2 (k) = n2 ;

    % Try MATLAB
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
    fprintf ('matlab: %12.6f GrB: ', t) ;
    T_matlab (k) = t ;

    A = GrB (A) ;

    T_grb (k,:) = inf (1,4) ;

    % Try each @GrB method
    for method = 1:4

        skip = false ;
        if (method == 1)
            % use GB_builder
            GB_mex_hack (1) ;
        elseif (method == 2)
            % use bucket: non-atomic
            if (nz < m)
                fprintf ('      -      ') ;
                continue ;
            end
            GB_mex_hack (-2) ;
        elseif (method == 3)
            % use bucket: atomic
            GB_mex_hack (-3) ;
        else
            % auto
            GB_mex_hack (0) ;
        end

        % Try GrB
        C = A' ;        % warmup
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
        fprintf ('%12.6f ', t) ;
        T_grb (k, method) = t ;
    end

    t_auto = T_grb (k, 4) ;
    t_matlab = T_matlab (k) ;

    rel = t_auto / min (T_grb (k,:)) ;
    if (rel == 1)
        fprintf ('auto best    ') ;
    elseif (rel < 1.1)
        fprintf ('auto ~       ') ;
    else
        fprintf ('auto %8.4f', rel) ;
    end

    fprintf (' MATLAB/GrB: %8.4f\n', t_matlab / t_auto) ;

    save tss_results list T_matlab T_grb Stats_nz Stats_m Stats_n Stats_m2 Stats_n2 

end

