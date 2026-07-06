function C = gzb_full (ghb, arg1, arg2, arg3, arg4)
%GZB_FULL: wrapper for gbmex_full mexFunction.  Not user-callable.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (gb_is_grb (arg1))
    arg1 = struct (arg1) ;
end

if (nargin >= 3 && gb_is_grb (arg2))
    arg2 = struct (arg2) ;
end

if (nargin >= 4 && gb_is_grb (arg3))
    arg3 = struct (arg3) ;
end

if (nargin >= 5 && gb_is_grb (arg4))
    arg4 = struct (arg4) ;
end

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

