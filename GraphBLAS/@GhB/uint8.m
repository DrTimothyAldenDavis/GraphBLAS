function C = uint8 (G)
%UINT8 cast a GraphBLAS matrix to built-in full uint8 matrix.
% C = uint8 (G) typecasts the GhB matrix G to a built-in full uint8
% matrix.  The result C is full since sparse uint8 matrices are not
% built-in.
%
% To typecast the matrix G to a GraphBLAS sparse uint8 matrix instead, use
% C = GhB (G, 'uint8').
%
% See also GhB, GhB/double, GhB/complex, GhB/single, GhB/logical, GhB/int8,
% GhB/int16, GhB/int32, GhB/int64, GhB/uint16, GhB/uint32, GhB/uint64.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

C = gb_cast_full (1, G, 'uint8', uint8 (0)) ;

