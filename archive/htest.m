function htest (A, B, nthreads)
% TODO delete when done
if (nargin > 2)
    GrB.threads (nthreads) ;
end
fprintf ('\nnthreads is %d\n', GrB.threads) ;
fprintf ('MATLAB:\n') ;
tic ; C1 = A*B ; toc
fprintf ('GrB:\n') ;
tic ; C2 = GrB.mxm (A, '+.*', B) ; toc

fprintf ('\n--------------------- GrB.hash3:\n') ;
tic ; C5 = GrB.hash3 (A,B) ; toc
err5 = norm (C1-C5,1) / norm (C1,1) ;

fprintf ('\n::::: GrB: min-times\n') ;
tic ; Ca = GrB.mxm (A, 'min.*', B) ; toc

fprintf ('\n--------------------- GrB.hash3min:\n') ;
tic ; Cb = GrB.hash3min (A, B) ; toc
errb = norm (Ca-Cb,1) / norm (Ca,1) ;

n = size (A, 1) ;
fprintf ('totnz/n %10.2f nnz C: %g A: %g B: %g\n', ...
    (nnz (C1) + nnz (A) + nnz (B))/n, ...
    nnz (C1), nnz (A), nnz (B)) ;
fprintf ('err %g %g %g\n\n', norm (C1-C2,1), err5, errb) ;
if (err5 > 1e-12 || err5 > 1e-12)
    error ('ack here!')
end

