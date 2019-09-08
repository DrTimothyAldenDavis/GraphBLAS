function e = nzmax (G)
%NZMAX the number of entries in a GraphBLAS matrix.
% Since the GraphBLAS data structure is opaque, nzmax (G) has no
% particular meaning.  Thus, nzmax (G) is simply max (gb.nvals (G), 1).  
% It appears as an overloaded operator for a gb matrix simply for
% compatibility with MATLAB sparse matrices.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

e = max (gbnvals (G.opaque), 1) ;

