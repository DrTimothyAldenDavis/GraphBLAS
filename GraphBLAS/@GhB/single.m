function C = single (G)
%SINGLE cast a GraphBLAS matrix to built-in full single matrix.
% C = single (G) typecasts the GhB matrix G to a built-in full single
% matrix.  The result C is full since sparse single matrices are not
% built-in.  C is real if G is real, and complex if G is complex.
%
% To typecast the matrix G to a GraphBLAS sparse single matrix instead,
% use C = GhB (G, 'single').  To typecast to a sparse single complex
% matrix, use G = GhB (G, 'single complex').
%
% See also GhB, GhB/double, GhB/complex, GhB/logical, GhB/int8, GhB/int16,
% GhB/int32, GhB/int64, GhB/uint8, GhB/uint16, GhB/uint32, GhB/uint64.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

ghb = 1 ;     % 0 for GrB, 1 for GhB

C = gb_single (ghb, G) ;

