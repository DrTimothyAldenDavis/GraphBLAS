function C = gzb_reduce (ghb, op, A)
%GZB_REDUCE: wrapper for gbmex_reduce mexFunction.  Not user-callable.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (gb_is_grb (A))
    A = struct (A) ;
end

if (ghb)
    C = GhB (gbmex_reduce (ghb, op, A)) ;
else
    C = GrB (gbmex_reduce (ghb, op, A)) ;
end

