function C = gzb_deserialize (ghb, blob)
%GZB_DESERIALIZE: wrapper for gbmex_deserialize mexFunction.  Not user-callable.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (ghb)
%   C = GhB (gbmex_deserialize (ghb, blob)) ;    % FIXME
else
    C = GrB (gbmex_deserialize (ghb, blob)) ;
end

