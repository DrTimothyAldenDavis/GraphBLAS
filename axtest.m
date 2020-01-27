
clear all
GrB.burble (1) ;
max_nthreads = 8 ;
threads = 1:max_nthreads ;
threads = 1 ;

n = 1e6 ;
nz = 50e6 ;
d = nz / n^2 ;
A = sprand (n,n,d) ;

ntrials = 10 ;

for test = 1 % 1:3

    if (test == 1)
        X = 'sparse (rand (n,1))' ;
        x =  sparse (rand (n,1)) ;
    elseif (test == 2)
        X = 'rand (n,1)' ;
        x =  rand (n,1) ;
    else
        X = 'sprand (n,1,0.05)' ;
        x =  sprand (n,1,0.05) ;
    end

    fprintf ('\n\n========================\n') ;
    fprintf ('in MATLAB: y = y + A*x where x = %s\n', X) ;

    for nthreads = threads
        maxNumCompThreads (nthreads) ;
        % y = sparse (n,1) ;
        y = sparse (ones (n,1)) ;
        tic
        for trial = 1:ntrials
            y = y + A*x ;
        end
        tmatlab (nthreads) = toc ;
        fprintf (...
            'threads: %2d MATLAB time: %8.4f sec speedup: %8.2f\n', ...
            nthreads, tmatlab (nthreads), tmatlab (1) / tmatlab (nthreads)) ;
        ymatlab = y ;
    end

    fprintf ('\ny = y + A*x where x = %s\n', X) ;

    A = GrB (A) ;
    x = GrB (x) ;
    for nthreads = threads
        GrB.threads (nthreads) ;
        tic
        % y = GrB (n,1) ;
        y = GrB (ones (n,1)) ;
        for trial = 1:ntrials
            y = y + A*x ;
        end
        t = toc ;
        if (nthreads == 1)
            t1 = t ;
        end
        fprintf (...
            'threads: %2d GrB time: %8.4f speedup vs MATLAB: %8.2f  vs: GrB(1 thread) %8.2f\n', ...
            nthreads, t, tmatlab(nthreads) / t, t1 / t) ;
        assert (norm (y-ymatlab, 1) / norm (ymatlab,1) < 1e-12)
    end


    fprintf ('\ny += A*x where x = %s\n', X) ;
    for nthreads = threads
        GrB.threads (nthreads) ;
        tic
        % y = GrB (n,1) ;
        y = GrB (ones (n,1)) ;
        for trial = 1:ntrials
            y = GrB.mxm (y, '+', '+.*', A, x) ;
        end
        t = toc ;
        if (nthreads == 1)
            t1 = t ;
        end
        fprintf (...
            'threads: %2d GrB time: %8.4f speedup vs MATLAB: %8.2f  vs: GrB(1 thread) %8.2f\n', ...
            nthreads, t, tmatlab(nthreads) / t, t1 / t) ;
        assert (norm (y-ymatlab, 1) / norm (ymatlab,1) < 1e-12)
    end

end
