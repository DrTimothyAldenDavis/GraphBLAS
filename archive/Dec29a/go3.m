
try
    GrB.finalize
catch
end
clear all
GrB.init
rng ('default') ;

n = 1e6 ;
nz = 3e6 ;
d = nz / n^2 ;
A = sprand (n, n, d) + speye (n) ;

fprintf ('MATLAB:\n') ;
tic ; C1 = A*A ; toc
for k = 0:10
    nth = 2^k ;
    GrB.threads (nth) ;
    fprintf ('\n=====================================  nthreads: %d\n', nth) ;
    tic ; [C3,h] = gbhash (A,A) ; toc
    err = norm (C1-C3,1) / norm (C1,1) ;
    if (err > 1e-12)
        err
        error ('ack!') ;
    end
end

B = sprand (n, 1, 0.1) ;
fprintf ('MATLAB:\n') ;
tic ; C1 = A*B ; tmatlab = toc

for k = 0:10
    nth = 2^k ;
    GrB.threads (nth) ;
    fprintf ('\n=====================================  nthreads: %d\n', nth) ;
    tic ; [C3,h] = gbhash (A,B) ; toc
    err = norm (C1-C3,1) / norm (C1,1) ;
    if (err > 1e-12)
        err
        error ('ack!') ;
    end
end

tmatlab
