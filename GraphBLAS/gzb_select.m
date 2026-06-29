function C = gzb_select (ghb, arg1, arg2, arg3)
%GZB_SELECT: wrapper for gbmex_select mexFunction.  Not user-callable.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (ghb)
    switch (nargin)
        case 3
            C = GhB (gbmex_select (ghb, arg1, arg2)) ;
        case 4
            C = GhB (gbmex_select (ghb, arg1, arg2, arg3)) ;
        otherwise
            error ('GrB:error', 'internal error 888') ;
    end
else
    switch (nargin)
        case 3
            C = GrB (gbmex_select (ghb, arg1, arg2)) ;
        case 4
            C = GrB (gbmex_select (ghb, arg1, arg2, arg3)) ;
        otherwise
            error ('GrB:error', 'internal error 888') ;
    end
end

