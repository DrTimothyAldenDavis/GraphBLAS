function C = gzb_emult (ghb, arg1, arg2, arg3, desc)
%GZB_EMULT: wrapper for gbmex_emult mexFunction.  Not user-callable.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (nargin < 5)
    desc = struct ;
end

if (ghb)
%   C = GhB (gbmex_emult (ghb, arg1, arg2, arg3, desc)) ;    % FIXME
else
    C = GrB (gbmex_emult (ghb, arg1, arg2, arg3, desc)) ;
end

