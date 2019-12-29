

try
    GrB.finalize
catch
end
clear all
GrB.init
rng ('default') ;

GrB.threads (4) ;

% n = 6 ;
% nz = 100 ;

desc = struct ;
% desc.kind = 'sparse' ;


Prob = ssget ('LAW/indochina-2004') ;
A = Prob.A  ;
n = size (A,1) ;
B = sparse (rand (n,1)) ;

for nth = [1 2 4 8]
    fprintf ('\nnthreads is %d\n', nth) ;
    GrB.threads (nth) ;
    tic ; C1 = A*B ; toc
    tic ; C2 = GrB.mxm (A, '+.*', B, desc) ; toc
    tic ; C3 = GrB.hash (A,B) ; toc
    C3 = (C3')' ;
    err2 = norm (C1-C3,1) / norm (C1,1) ;
    fprintf ('totnz/n %10.2f\n', (nnz (C1) + nnz (A) + nnz (B))/n) ;
    fprintf ('err1 %g err2 %g\n\n', norm (C1-C2,1), err2) ;
    if (err2 > 1e-12)
        error ('ack!')
    end
end


B = sprandn (n, 1, 0.01) ;

for nth = [1 2 4 8]
    fprintf ('\nnthreads is %d\n', nth) ;
    GrB.threads (nth) ;
    tic ; C1 = A*B ; toc
    tic ; C2 = GrB.mxm (A, '+.*', B, desc) ; toc
    tic ; C3 = GrB.hash (A,B) ; toc
    C3 = (C3')' ;
    nnz (C3)
    err2 = norm (C1-C3,1) / norm (C1,1) ;
    fprintf ('totnz/n %10.2f\n', (nnz (C1) + nnz (A) + nnz (B))/n) ;
    fprintf ('err1 %g err2 %g\n\n', norm (C1-C2,1), err2) ;
    if (err2 > 1e-12)
        error ('ack!')
    end
end
