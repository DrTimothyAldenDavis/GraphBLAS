function result = istriu (G)
%ISTRIU  Determine if a matrix is upper triangular.
% A GraphBLAS matrix G may have explicit zeros.  If these appear in the
% lower triangular part of G, then istriu (G) is false, but
% istriu (double (G)) can be true since the double (G) drops those entries.

% FUTURE: this will be much faster when written as a mexFunction.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

result = (gb.nvals (tril (G, -1)) == 0) ;

