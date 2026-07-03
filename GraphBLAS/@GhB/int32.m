function C = int32 (G)
%INT32 cast a GraphBLAS matrix to built-in full int32 matrix.
% C = int32 (G) typecasts the GhB matrix G to a built-in full int32 matrix.
% The result C is full since sparse int32 matrices are not built-in.
%
% To typecast the matrix G to a GraphBLAS sparse int32 matrix instead,
% use C = GhB (G, 'int32').
%
% See also GhB, GhB/double, GhB/complex, GhB/single, GhB/logical, GhB/int8,
% GhB/int16, GhB/int64, GhB/uint8, GhB/uint16, GhB/uint32, GhB/uint64.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

ghb = 1 ;     % 0 for GrB, 1 for GhB

desc.kind = 'full' ;
C = gbmex_builtin (gzb_full (ghb, G, 'int32', int32 (0), desc)) ;

