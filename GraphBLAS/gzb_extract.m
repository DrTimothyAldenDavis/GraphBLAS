function C = gzb_extract (ghb, A, I, J)
%GZB_EXTRACT: wrapper for gbmex_extract mexFunction.  Not user-callable.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (ghb)
%   C = GhB (gbmex_extract (ghb, A, I, J)) ;    FIXME
else
    C = GrB (gbmex_extract (ghb, A, I, J)) ;
end

