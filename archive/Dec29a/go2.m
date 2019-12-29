
try
    GrB.finalize
catch
end
clear all
GrB.init
rng ('default') ;

GrB.threads (2) ;

% n = 10000 ;
% nz = 1e6 ;
n = 20 ;
nz = 10 ;
d = nz / n^2 ;

desc.kind = 'sparse' ;

A = sprand (n,n,d) ;
B = sprand (n,n,d) ;

GrB.threads (2) ;

B (:,20) = 1 ;

tic ; C1 = A*B ; toc
tic ; C2 = GrB.mxm (A, '+.*', B, desc) ; toc
tic ; [C3,h] = gbhash (A,B) ; toc
fprintf ('totnz/n %10.2f\n', (nnz (C1) + nnz (A) + nnz (B))/n) ;
fprintf ('h %g err1 %g err2 %g\n\n', h, norm (C1-C2,1), norm (C1-C3,1)) ;

