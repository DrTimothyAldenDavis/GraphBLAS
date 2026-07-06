function C = double (G)
%DOUBLE cast a GraphBLAS matrix to a built-in double matrix.
% C = double (G) typecasts the GraphBLAS matrix G into a built-in
% double matrix C, either real or complex.  C is full if all
% entries in G are present, and sparse otherwise.
%
% To typecast the matrix G to a GraphBLAS double (real) matrix
% instead, use C = GhB (G, 'double').  Explicit zeros are kept in C.
%
% See also GhB/cast, GhB, GhB/complex, GhB/single, GhB/logical, GhB/int8,
% GhB/int16, GhB/int32, GhB/int64, GhB/uint8, GhB/uint16, GhB/uint32,
% GhB/uint64.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

C = gb_double (1, G) ;

