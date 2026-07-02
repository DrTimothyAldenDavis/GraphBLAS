function C = gzb_full (ghb, arg1, arg2, arg3, arg4)
%GZB_FULL: wrapper for gbmex_full mexFunction.  Not user-callable.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (ghb)
    switch (nargin)
        case 2
            C = GhB (gbmex_full (ghb, arg1)) ;
        case 3
            C = GhB (gbmex_full (ghb, arg1, arg2)) ;
        case 4
            C = GhB (gbmex_full (ghb, arg1, arg2, arg3)) ;
        case 5
            C = GhB (gbmex_full (ghb, arg1, arg2, arg3, arg4)) ;
    end
else
    switch (nargin)
        case 2
            C = GrB (gbmex_full (ghb, arg1)) ;
        case 3
            C = GrB (gbmex_full (ghb, arg1, arg2)) ;
        case 4
            C = GrB (gbmex_full (ghb, arg1, arg2, arg3)) ;
        case 5
            C = GrB (gbmex_full (ghb, arg1, arg2, arg3, arg4)) ;
    end
end

