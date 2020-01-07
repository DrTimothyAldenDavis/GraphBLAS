
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

C-C2

