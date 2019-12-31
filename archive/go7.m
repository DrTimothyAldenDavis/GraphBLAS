% TODO move to Test, or delete

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

n = 1e6 ;
m = n ;
d = 1e-5 ;
A = speye (n) + sprand (n, n, d) ;
B = A ;

for nth = threads
    fprintf ('\nm %d =======================threads is %d nnz(B) is %d\n', ...
        m, nth, nnz(B)) ;
    htest (A, B, nth) ;
end

