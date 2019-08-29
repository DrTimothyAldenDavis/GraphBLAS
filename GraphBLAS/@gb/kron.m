function C = kron (A, B)
%KRON sparse Kronecker product
% C = kron (A,B) is the sparse Kronecker tensor product of A and B.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

C = gb.gbkron ('*', A, B) ;

