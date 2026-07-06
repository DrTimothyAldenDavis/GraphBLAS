function C = int64 (G)
%INT64 cast a GraphBLAS matrix to built-in full int64 matrix.
% C = int64 (G) typecasts the GhB matrix G to a full int64 matrix.  The
% result C is full since sparse int64 matrices are not built-in.
%
% To typecast the matrix G to a GraphBLAS sparse int64 matrix instead,
% use C = GhB (G, 'int64').
%
% See also GhB, GhB/double, GhB/complex, GhB/single, GhB/logical, GhB/int8,
% GhB/int16, GhB/int32, GhB/uint8, GhB/uint16, GhB/uint32, GhB/uint64.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

C = gb_cast_full (1, G, 'int64', int64 (0)) ;

