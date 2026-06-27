function C = gzb_serialize (ghb, A, method, level)
%GZB_SERIALIZE: wrapper for gbmex_serialize mexFunction.  Not user-callable.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (ghb)
    % FIXME
else
    switch (nargin)
        case 2
            C = GrB (gbmex_serialize (ghb, A)) ;
        case 3
            C = GrB (gbmex_serialize (ghb, A, method)) ;
        case 4
            C = GrB (gbmex_serialize (ghb, A, method, level)) ;
        otherwise
            error ('GrB:error', 'internal error 887') ;
    end
end

