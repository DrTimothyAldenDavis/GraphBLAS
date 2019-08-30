function s = numel (G)
%NUMEL the maximum number of entries a GraphBLAS matrix can hold.
% numel (G) is m*n for the m-by-n GraphBLAS matrix G.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

[m, n] = size (G) ;
s = m*n ;

