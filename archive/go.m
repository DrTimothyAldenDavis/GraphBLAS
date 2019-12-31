% TODO move to Test, or delete

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

n = 10000 ;
nz = 1e6 ;
% n = 6 ;
% nz = 100 ;
d = nz / n^2 ;

A = sprand (n,n,d) ;
B = sprand (n,n,d) ;

htest (A,B) ;

B (:,end) = 1 ;
for nth = threads
    htest (A, B, nth) ;
end

B = sprand (n,n, 1e-4) + speye (n) ;
htest (A, B, 8) ;

B (:,end) = 1 ;

htest (A, B, 8) ;

B = sparse (rand (n,1)) ;

htest (A, B, 8) ;

B = sprand (n,10,d) ;
B (:,3) = 1 ;

htest (A, B, 8) ;

Prob = ssget (2662) ;
A = Prob.A ;
n = size (A, 1) ;

fprintf ('\n========== A is Freescale, b is n-by-2 and dense\n') ;
B = sparse (rand (n,2)) ;
for nth = threads
    htest (A, B, nth) ;
end

fprintf ('\n========== A is Freescale, b is n-by-2 and sparse\n') ;
B = sprand (n,2, 0.1) ;
for nth = threads
    if (nth > nthreads_max)
        break ;
    end
    htest (A, B, nth) ;
end

fprintf ('=================== A is Freescale, b is n-by-2 and very sparse\n') ;
B = sprand (n,2, 0.01) ;
for nth = threads
    htest (A, B, nth) ;
end

fprintf ('=================== A is Freescale, b is almost diagonal\n') ;
B = 2 * speye (n) ;
B(1,2) = 1 ;
for nth = threads
    htest (A, B, nth) ;
end

fprintf ('\n==== A is Freescale + dense col, b is n-by-10\n') ;
B = 2 * speye (n,10) ;
B(1,2) = 1 ;
B (2,1) = 1 ;
A (:,1) = 1 ;
for nth = threads
    htest (A, B, nth) ;
end

fprintf ('\n==== A is Freescale + almost dense col, b is n-by-10\n') ;
A (1,1:2) = 1 ;
for nth = threads
    htest (A, B, nth) ;
end

