function htest (A, B, nthreads)
if (nargin > 2)
    GrB.threads (nthreads) ;
end
fprintf ('\nnthreads is %d\n', GrB.threads) ;
fprintf ('MATLAB:\n') ;
tic ; C1 = A*B ; toc
fprintf ('GrB:\n') ;
tic ; C2 = GrB.mxm (A, '+.*', B) ; toc
fprintf ('\n--------------------- GrB.hash:\n') ;
tic ; C3 = GrB.hash (A,B) ; toc
% tic ; C4 = GrB.hash2 (A,B) ; toc
fprintf ('\n--------------------- GrB.hash3:\n') ;
tic ; C5 = GrB.hash3 (A,B) ; toc
% tic ; C6 = GrB.hash4 (A,B) ; toc
err2 = norm (C1-C3,1) / norm (C1,1) ;
% err4 = norm (C1-C4,1) / norm (C1,1) ;
err5 = norm (C1-C5,1) / norm (C1,1) ;
% err6 = norm (C1-C6,1) / norm (C1,1) ;
n = size (A, 1) ;
fprintf ('totnz/n %10.2f nnz C: %g A: %g B: %g\n', ...
    (nnz (C1) + nnz (A) + nnz (B))/n, ...
    nnz (C1), nnz (A), nnz (B)) ;
fprintf ('err %g %g g %g\n\n', norm (C1-C2,1), err2, err5) ;
if (err2 > 1e-12 || err5 > 1e-12)
    error ('ack here!')
end

