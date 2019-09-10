% clear all
Prob = ssget ('HB/ibm32') ;
A = Prob.A ;
H = gb (A)

S = A+A' ;
H2 = gb (S)

disp (H2,3)

H3 = graph (S)

L2 = gb.laplacian (H2)
L3 = laplacian (graph (S, 'OmitSelfLoops')) ;

assert (isequal (L3, double (L3))) ;
