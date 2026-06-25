function C = gzb_loadhistorical (ghb, S)
%GZB_LOADHISTORICAL: wrapper for gbmex_loadhistorical mexFunction.
% Not user callable.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (ghb)
%   C = GhB (gbmex_loadhistorical (ghb, S)) ;    FIXME
else
    C = GrB (gbmex_loadhistorical (ghb, S)) ;
end

