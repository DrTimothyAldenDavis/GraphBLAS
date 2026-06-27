function C = gzb_kronecker (ghb, A, op, B)
%GZB_KRONECKER: wrapper for gbmex_kronecker mexFunction.  Not user-callable.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (ghb)
%   C = GhB (gbmex_kronecker (ghb, A, op, B)) ;    FIXME
else
    C = GrB (gbmex_kronecker (ghb, A, op, B)) ;
end

