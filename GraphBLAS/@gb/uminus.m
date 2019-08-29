function C = uminus (A)
%UMINUS negate a GraphBLAS sparse matrix.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

C = gb.apply ('-', A) ;

