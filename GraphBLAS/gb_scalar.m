function x = gb_scalar (A)
%GB_SCALAR get contents of a scalar.  Not user-callable.
% A may be a built-in scalar or a GraphBLAS scalar.  Returns the result
% x as a built-in non-sparse scalar.  If the scalar has no entry (the
% built-in sparse(0)), then x is returned as zero.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (gb_is_grb (A))
    A = struct (A) ;
end

gbmex_wait (A) ;
[~, ~, x] = gbmex_extracttuples (1, A) ;
if (isempty (x))
    x = 0 ;
else
    x = x (1) ;
end

