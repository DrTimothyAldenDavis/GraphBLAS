
clear all
tic
n = 2^29 ;
nz = 2000 ;

i = ceil (n * rand (nz, 1)) ;
j = ceil (n * rand (nz, 1)) ;
x = rand (nz, 1) ;
A = gb.build (i, j, x, n, n) 

i = ceil (n * rand (nz, 1)) ;
j = ceil (n * rand (nz, 1)) ;
x = rand (nz, 1) ;
B = gb.build (i, j, x, n, n) 

C = gb.gbkron ('*', A, B) 

[i j x] = gb.extracttuples (C) ;
s = sum (C) 
H = C' 
H (1:3, 1:3) = magic (3)
toc

% make the pattern unsymmetric.
% H(1,:) is already non-empty, so 
H (1,99) = 42 ;

tic
rowdegree = sum (spones (H')) ;
non_empty_rows = nnz (rowdegree > 0) 

coldegree = sum (spones (H)) ;
non_empty_cols = nnz (coldegree > 0) 
whos


toc
