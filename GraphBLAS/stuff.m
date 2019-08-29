% clear all
Prob = ssget ('HB/ibm32') ;
A = Prob.A ;
H = gbgraph (A)

S = A+A' ;
H2 = gbgraph (S)

disp (H2,3)

H3 = graph (S)

L2 = laplacian (H2)
L3 = laplacian (graph (S, 'OmitSelfLoops'))

assert (isequal (L3, double (L3))) ;
