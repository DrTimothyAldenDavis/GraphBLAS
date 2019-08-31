function C = kron (A, B)
%KRON sparse Kronecker product.
% C = kron (A,B) is the sparse Kronecker tensor product of A and B.
% The input matrices may be either GraphBLAS and/or MATLAB matrices, in
% any combination.  C is returned as a GraphBLAS matrix.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

C = gb.gbkron ('*', A, B) ;

