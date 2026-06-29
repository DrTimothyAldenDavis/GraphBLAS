function C = gzb_build (ghb, I, J, X, m, n, arg7, arg8, arg9)
%GZB_BUILD: wrapper for gbmex_build mexFunction.  Not user-callable.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (ghb)
    switch (nargin)
        case 7
            C = GhB (gbmex_build (ghb, I, J, X, m, n, arg7)) ;
        case 8
            C = GhB (gbmex_build (ghb, I, J, X, m, n, arg7, arg8)) ;
        case 9
            C = GhB (gbmex_build (ghb, I, J, X, m, n, arg7, arg8, arg9)) ;
        otherwise
            error ('GrB:error', 'internal error 889') ;
    end
else
    switch (nargin)
        case 7
            C = GrB (gbmex_build (ghb, I, J, X, m, n, arg7)) ;
        case 8
            C = GrB (gbmex_build (ghb, I, J, X, m, n, arg7, arg8)) ;
        case 9
            C = GrB (gbmex_build (ghb, I, J, X, m, n, arg7, arg8, arg9)) ;
        otherwise
            error ('GrB:error', 'internal error 889') ;
    end
end

