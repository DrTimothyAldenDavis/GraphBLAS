

try
    GrB.finalize
    clear all
    GrB.init
catch me
    me
end
rng ('default')

Prob = ssget (2451)
A = GrB (Prob.A, 'single') ;
[m n] = size (A) ;
B = GrB (rand (n,1), 'single') ;
S = GrB (m, 1, 'single') ;
A (:,1:3) = 1 ;
B (1:3) = 1 ;

for e = 1:32
    nz = 2^e ;
    if (nz > m)
        break ;
    end
    M = GrB (m, 1, 'logical') ;
    M (randperm (m, nz)) = true ;
    fprintf ('e: %2d  M/m %8.3e ', e, nnz (M) / m) ;

    tic ;
    C = GrB.mxm (S, M, '+.2nd.single', A, B) ;
    t1 = toc  ;
    fprintf (' %10.4f  ', t1) ;

    tic ;
    T = GrB.mxm ('+.2nd.single', A, B) ;
    C = GrB.assign (S, M, T) ;
    t2 = toc  ;
    fprintf (' %10.4f  rel: t2/t1 %g\n', t2, t2/t1) ;


end

