function result = istril (G)
%ISTRIL  Determine if a matrix is lower triangular.
% A GraphBLAS matrix G may have explicit zeros.  If these appear in the
% upper triangular part of G, then istril (G) is false, but
% istril (double (G)) can be true since double (G) drops those entries.

% FUTURE: this will be much faster when written as a mexFunction.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

result = (gb.nvals (triu (G, 1)) == 0) ;

