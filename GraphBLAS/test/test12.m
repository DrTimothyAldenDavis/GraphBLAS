clear all

rng ('default') ;
A = sparse (rand (2)) ;
B = sparse (rand (2)) ;

C = A+B ;
D = A.*B ;

G = gb.eadd ('+', A, B) ;
err = norm (C-sparse(G), 1)
assert (err < 1e-12)

H = gb.emult ('*', A, B) ;
err = norm (D-sparse(H), 1)
assert (err < 1e-12)

d.kind = 'sparse' ;
d.in0 = 'transpose' ;
d

G = gb.eadd ('+', A, B, d) ;
C = A'+B ;
err = norm (C-sparse(G), 1)
assert (err < 1e-12)

H = gb.emult ('*', A, B, d) ;
D = A'.*B ;
err = norm (H-sparse(D), 1)
assert (err < 1e-12)

d.kind = 'gb' ;
G = gb.eadd ('+', A, B, d) ;
G = sparse (G) ;
err = norm (C-G, 1)

H = gb.emult ('*', A, B, d) ;
H = sparse (H) ;
err = norm (D-H, 1)

E = sparse (rand (2)) ;
C = E + A+B ;
G = gb.eadd (E, '+', '+', A, B) ; 
C-sparse(G)

F = sparse (rand (2)) ;
D = F + A.*B ;
H = gb.emult (F, '+', '*', A, B) ; 
D-sparse(H)

G = gb.eadd ('+', A, B)
C = A+B

H = gb.emult ('*', A, B)
D = A.*B

