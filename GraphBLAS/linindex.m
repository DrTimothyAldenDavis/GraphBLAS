
gb.format ('by row') ;
n = 10
m = 6
A = sprand (m, n, 0.5)
mn = m*n 
G = reshape (gb (A), 5, 12)
S = reshape (A, 5, 12)
assert (isequal (S, double (G)))

A (2:10)
F = full (A)

I = 2:10
x = F (2:10)
x = reshape (x, size (I))


x = F (2:10) ;
x = x (:) ;
x = reshape (x, size (I))



gb.format ('by col') ;
