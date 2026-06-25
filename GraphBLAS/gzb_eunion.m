function C = gzb_eunion (ghb, arg1, arg2, arg3, arg4, arg5)
%GZB_EUNION: wrapper for gbmex_eunion mexFunction.  Not user callable.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (ghb)
%   C = GhB (gbmex_eunion (ghb, arg1, arg2, arg3, arg4, arg5)) ;    % FIXME
else
    C = GrB (gbmex_eunion (ghb, arg1, arg2, arg3, arg4, arg5)) ;
end

