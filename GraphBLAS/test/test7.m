clear all

n = 5 ;
A = sprand (n, n, 0.5) ;

[i j x] = find (A) ;
[m n] = size (A) ;

G = gb.build (i, j, x, m, n)
S = sparse   (i, j, x, m, n)

sparse (G)-S

d.kind = 'gb' ;
G = gb.build (i, j, x, m, n, d) ;
S - sparse (G)

G = gb.build (i, j, x, m, n, d)
S - sparse (G)

