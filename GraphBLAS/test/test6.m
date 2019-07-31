clear all

rng ('default') ;
A = sparse (rand (2)) ;
B = sparse (rand (2)) ;

C = A*B ;

G = gbmxm ('+.*', A, B) ;
err = norm (C-G, 1)
assert (err < 1e-12)

d.kind = 'sparse' ;
d.in0 = 'transpose' ;
d
G = gbmxm ('+.*', A, B, d) ;
C = A'*B ;

err = norm (C-G, 1)
assert (err < 1e-12)

d.kind = 'object' ;
G = gbmxm ('+.*', A, B, d) ;
G = sparse (gb (G)) ;
err = norm (C-G, 1)

E = sparse (rand (2)) ;
C = E + A*B ;
G = gbmxm (E, '+', '+.*', A, B) ; 
C-G

G = gb.mxm ('+.*', A, B)

