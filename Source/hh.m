
try
    GrB.finalize
    clear all
    GrB.init
catch me
    me
end
rng ('default')

rng ('default')
n = 4 ;
A = GrB (rand (n))
B = GrB (rand (n))
S = GrB (sparse (n,n)) ;
M = (A > 0.5)

C = GrB.mxm (S, M, '+.*', A, B)
C = full (double (C))

T = A*B 
C2 = S ;
C2 (M) = T (M) ;
C2 = full (double (C2))
T = full (double (T))

C-C2


m = 256 * 1024 ;
n = 64 ;

% A = rand (m,n) ;
% B = rand (n,n) ;
% tic ; C = A*B ;toc ;

A = sprand (m, n, 0.5) ;
B = sprand (n, n, 0.5) ;
tic ; C = A*B ;toc ;

A = GrB (A) ;
B = GrB (B) ;

for nthreads = 1:4
    GrB.threads (nthreads) ;
    fprintf ('GrB: %d\n', nthreads) ;
    tic ; C = A*B ;toc ;
end

S = GrB (m,n) ;
anz = nnz (A) ;
d = anz / n ;

for e = 1:16
    nz = 2^e ;
    fprintf ('e %2d : %10g \n', e, d/nz) ;
    M = logical (sparse (m,n)) ;
    for k = 1:n
        p = randperm (m, nz) ;
        M (p,k) = true ;
    end
    for nthreads = 1:4
        fprintf ('   %d: ') ;
        GrB.threads (nthreads) ;
        tic ;
        C = GrB.mxm (S, M, '+.*', A, B) ;
        t = toc ;
        fprintf ('%8.3f ', t) ;

        tic ;
        % C2 = S ;
        T = A*B ;
        % C2 (M) = T (M) ;
        C2 = GrB.assign (S, M, T) ;
        t = toc ;
        fprintf ('%8.3f ', t) ;
        
        err = norm (C-C2,1) / norm (C,1) ;
        % fprintf ('%8.3e ', err) ;
        fprintf ('\n') ;
        if (err > 1e-10)
            error ('!') 
        end

    end
    fprintf ('\n') ;
end
