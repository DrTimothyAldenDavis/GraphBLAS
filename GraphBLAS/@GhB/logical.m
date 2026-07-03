function C = logical (G)
%LOGICAL typecast a GraphBLAS matrix to built-in logical matrix.
% C = logical (G) typecasts the GraphBLAS matrix G to into a built-in
% logical matrix.  C is full if all entries in G are present, and
% sparse otherwise.
%
% To typecast the matrix G to a GraphBLAS logical matrix instead,
% use C = GhB (G, 'logical').
%
% See also cast, GhB, GhB/double, GhB/complex, GhB/single, GhB/int8,
% GhB/int16, GhB/int32, GhB/int64, GhB/uint8, GhB/uint16, GhB/uint32,
% GhB/uint64.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

ghb = 1 ;     % 0 for GrB, 1 for GhB

C = gbmex_builtin (gzb_cast (ghb, G, 'logical')) ;

