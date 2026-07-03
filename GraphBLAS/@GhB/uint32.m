function C = uint32 (G)
%UINT32 cast a GraphBLAS matrix to built-in full uint32 matrix.
% C = uint32 (G) typecasts the GhB matrix G to a built-in full uint32
% matrix.  The result C is full since sparse uint32 matrices are not
% built-in.
%
% To typecast the matrix G to a GraphBLAS sparse uint32 matrix instead,
% use C = GhB (G, 'uint32').
%
% See also GhB, GhB/double, GhB/complex, GhB/single, GhB/logical, GhB/int8,
% GhB/int16, GhB/int32, GhB/int64, GhB/uint8, GhB/uint16, GhB/uint64.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

ghb = 1 ;     % 0 for GrB, 1 for GhB

desc.kind = 'full' ;
C = gbmex_builtin (gzb_full (ghb, G, 'uint32', uint32 (0), desc)) ;

