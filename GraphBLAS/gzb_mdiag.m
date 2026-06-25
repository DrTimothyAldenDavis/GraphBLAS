function C = gzb_mdiag (ghb, A, k)
%GZB_MDIAG: wrapper for gbmex_mdiag mexFunction. Not user callable.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (ghb)
%   C = GhB (gbmex_mdiag (ghb, A, k)) ;    FIXME
else
    C = GrB (gbmex_mdiag (ghb, A, k)) ;
end

