function C = gzb_eadd (ghb, arg1, arg2, arg3, desc)
%GZB_EADD: wrapper for gbmex_eadd mexFunction.  Not user-callable.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin < 5)
    desc = struct ;
end

if (ghb)
    C = GhB (gbmex_eadd (ghb, arg1, arg2, arg3, desc)) ;
else
    C = GrB (gbmex_eadd (ghb, arg1, arg2, arg3, desc)) ;
end

