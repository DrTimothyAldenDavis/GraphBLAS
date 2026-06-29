function C = gzb_apply (ghb, arg1, arg2, desc)
%GZB_APPLY: wrapper for gbmex_apply mexFunction.  Not user-callable.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin < 4)
    desc = struct ;
end

if (ghb)
    C = GhB (gbmex_apply (ghb, arg1, arg2, desc)) ;
else
    C = GrB (gbmex_apply (ghb, arg1, arg2, desc)) ;
end


