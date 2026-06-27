function x = gb_get_scalar (ghb, A)
%GB_GET_SCALAR get a scalar from a matrix.  Not user-callable.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

[m, n] = gbmex_size (A) ;
if (m ~= 1 || n ~= 1)
    error ('GrB:error', 'input parameter %s must be a scalar', inputname (1)) ;
end

x = gb_scalar (ghb, A) ;

