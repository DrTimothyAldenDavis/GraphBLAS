function C = gb_maxall (op, A)
%GB_MAXALL reduce a matrix to a scalar
% Implements C = max (A, [ ], 'all') ;

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

C = GrB (gbmex_reduce (op, A)) ;
if (~gb_isfull (A) && gb_scalar (C) <= 0)
    % A is not full, and the max of the entries present is <= 0,
    % so C is an empty scalar (an implicit zero)
    C = GrB (1, 1, gbmex_type (C)) ;
end

