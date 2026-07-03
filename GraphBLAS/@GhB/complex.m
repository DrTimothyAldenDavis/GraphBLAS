function C = complex (A, B)
%COMPLEX cast to a built-in double complex matrix.
% C = complex (G) typecasts the GraphBLAS matrix G to into a built-in
% double complex matrix.  C is full if all entries in G are present,
% or sparse otherwse.
%
% With two inputs, C = complex (A,B) returns a matrix C = A + 1i*B, where A or
% B are real matrices.  If A or B are nonzero scalars and the other input is a
% matrix, or if both A and B are scalars, C is full.
%
% To typecast the matrix G to a GraphBLAS double complex matrix
% instead, use C = GhB (G, 'complex') or C = GhB (G, 'double complex').
% To typecast the matrix G to a GraphBLAS single complex matrix, use
% C = GhB (G, 'single complex').
%
% To construct a complex GraphBLAS matrix from real GraphBLAS matrices
% A and B, use C = A + 1i*B instead.
%
% Since sparse single complex matrices are not built-in, C is
% always returned as a double complex matrix (sparse or full).
%
% See also cast, GhB, GhB/double, GhB/single, GhB/logical, GhB/int8,
% GhB/int16, GhB/int32, GhB/int64, GhB/uint8, GhB/uint16, GhB/uint32,
% GhB/uint64.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

ghb = 1 ;     % 0 for GrB, 1 for GhB

if (nargin == 1)
    C = gb_complex (ghb, A) ;
else
    C = gb_complex (ghb, A, B) ;
end

