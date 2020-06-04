function s = gb_isfull (A)
%GB_ISFULL determine if all entries are present in a GraphBLAS struct.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights
% Reserved. http://suitesparse.com.  See GraphBLAS/Doc/License.txt.

[m, n] = gbsize (A) ;
s = (m*n == gbnvals (A)) ;

