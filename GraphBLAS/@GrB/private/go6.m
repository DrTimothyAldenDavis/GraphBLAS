

try
    GrB.finalize
catch
end
clear all
GrB.init
rng ('default') ;

nthreads_max = 2 * GrB.threads ;
threads = [1 2 4 8 16 20 32 40 64] ;
threads = threads (threads <= nthreads_max) ;

% n = 6 ;
% nz = 100 ;

desc = struct ;
% desc.kind = 'sparse' ;


Prob = ssget ('LAW/indochina-2004') ;
A = Prob.A  ;
clear Prob ;
n = size (A,2) ;
B = spones (sprandn (n, 1, 0.1)) ;
m = size (A,1) ;

for nth = threads
    fprintf ('\nm %d =======================threads is %d nnz(B) is %d\n', ...
        m, nth, nnz(B)) ;
    htest (A, B, nth) ;
end

A (4e9,1) = 1 ;
A (4e9,1) = 0 ;
m = size (A,1) ;
fprintf ('# of rows of A: %g million\n', m / 1e6) ;
fprintf ('# of cols of A: %g million\n', n / 1e6) ;

for nth = threads
    fprintf ('\nm %d =======================threads is %d nnz(B) is %d\n', ...
        m, nth, nnz(B)) ;
    htest (A, B, nth) ;
end

