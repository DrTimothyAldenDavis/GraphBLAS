clear all

rng ('default') ;
A = sparse (rand (2)) ;
B = sparse (rand (2)) ;

C = A*B ;

G = gb.mxm ('+.*', A, B) ;
err = norm (C-sparse(G), 1)
assert (err < 1e-12)

d.kind = 'sparse' ;
d.in0 = 'transpose' ;
d
G = gb.mxm ('+.*', A, B, d) ;
C = A'*B ;

err = norm (C-sparse(G), 1)
assert (err < 1e-12)

d.kind = 'gb' ;
G = gb.mxm ('+.*', A, B, d) ;
G = sparse (G) ;
err = norm (C-G, 1)

E = sparse (rand (2)) ;
C = E + A*B ;
G = gb.mxm (E, '+', '+.*', A, B) ; 
C-sparse(G)

G = gb.mxm ('+.*', A, B)

