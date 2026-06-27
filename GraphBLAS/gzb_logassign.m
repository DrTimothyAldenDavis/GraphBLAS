function C = gzb_logassign (ghb, Cin, M, A)
%GZB_LOGASSIGN: wrapper for gbmex_logassign mexFunction.  Not user-callable.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (ghb)
%   C = GhB (gbmex_logassign (ghb, Cin, M, A)) ;    % FIXME
else
    C = GrB (gbmex_logassign (ghb, Cin, M, A)) ;
end

