function C = spfun (fun, G)
%SPFUN Apply function to the entries of a GraphBLAS matrix.
% C = spfun (fun, G) evaluates the function fun on the entries of G.

% FUTURE: this would be much faster as a mexFunction, but calling feval
% from inside a mexFunction would not be trivial.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

[m n] = size (G) ;
[i j x] = gb.extracttuples (G, struct ('kind', 'zero-based')) ;
x = feval (fun, x) ;
C = gb.build (i, j, x, m, n, '1st', gb.type (x)) ;

