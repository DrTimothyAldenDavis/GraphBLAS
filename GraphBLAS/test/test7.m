clear all

n = 5 ;
A = sprand (n, n, 0.5) ;

[i j x] = find (A) ;
[m n] = size (A) ;

G = gbbuild (i, j, x, m, n)
S = sparse  (i, j, x, m, n)

G-S

d.kind = 'object' ;
G = gbbuild (i, j, x, m, n, d) ;
S - sparse (gb (G))

G = gb.build (i, j, x, m, n, d)
S - sparse (G)

