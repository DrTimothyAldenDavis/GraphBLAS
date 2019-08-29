function e = nzmax (G)
%NZMAX the number of entries in a GraphBLAS matrix.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

e = max (gbnvals (G.opaque), 1) ;

