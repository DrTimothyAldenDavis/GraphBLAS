function C = gzb_mtimes (ghb, A, B)
%GZB_MTIMES: wrapper for gbmex_mtimes mexFunction. Not user callable.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (ghb)
%   C = GhB (gbmex_mtimes (ghb, A, B)) ;    FIXME
else
    C = GrB (gbmex_mtimes (ghb, A, B)) ;
end

