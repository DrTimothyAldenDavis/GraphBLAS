function C = kron (A, B)
%KRON sparse Kronecker product.
% C = kron (A,B) is the sparse Kronecker tensor product of A and B.
%
% See also GrB.kronecker.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

ghb = 1 ;     % 0 for GrB, 1 for GhB

C = GrB (gbmex_kronecker (ghb, A, '*', B)) ;

