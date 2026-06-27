function C = gzb_vreduce (ghb, arg1, arg2, arg3, arg4, arg5)
%GZB_VREDUCE: wrapper for gbmex_vreduce mexFunction.  Not user-callable.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (ghb)
    % FIXME
else
    switch (nargin)
        case 3
            C = GrB (gbmex_vreduce (ghb, arg1, arg2)) ;
        case 4
            C = GrB (gbmex_vreduce (ghb, arg1, arg2, arg3)) ;
        case 5
            C = GrB (gbmex_vreduce (ghb, arg1, arg2, arg3, arg4)) ;
        case 6
            C = GrB (gbmex_vreduce (ghb, arg1, arg2, arg3, arg4, arg5)) ;
        otherwise
            error ('GrB:error', 'internal error 886') ;
    end
end

