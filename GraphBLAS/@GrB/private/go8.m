

try
    GrB.finalize
catch
end
clear all
GrB.init
rng ('default') ;

% n = 6 ;
% nz = 100 ;

nthreads_max = 2 * GrB.threads ;
threads = [1 2 4 8 16 20 32 40 64] ;
threads = threads (threads <= nthreads_max) ;

desc = struct ;
% desc.kind = 'sparse' ;

matrices = ssgrep ('GAP') ;

for id = matrices
    
    fprintf ('\n################################################################################\n') ;
    tic
    Prob = ssget (id)
    toc
    A = Prob.A ;
    n = size (A,2) ;
    m = size (A,1) ;

    B = sparse (rand (n,1)) ;
    for nth = threads
        fprintf ('\nm %d =======================threads is %d nnz(B) is %d\n', ...
            m, nth, nnz(B)) ;
        htest (A, B, nth) ;
    end

    B = sprand (n, 1, 0.001) ;
    for nth = threads
        fprintf ('\nm %d =======================threads is %d nnz(B) is %d\n', ...
            m, nth, nnz(B)) ;
        htest (A, B, nth) ;
    end
end

