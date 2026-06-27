function d = gzb_degree (ghb, A, dim)
%GZB_DEGREE: wrapper for gbmex_degree mexFunction.  Not user-callable.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (ghb)
%   d = GhB (gbmex_degree (ghb, A, dim)) ;    % FIXME
else
    d = GrB (gbmex_degree (ghb, A, dim)) ;
end

