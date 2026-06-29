function C = gzb_cat (ghb, Tiles)
%GZB_CAT: wrapper for gbmex_cat mexFunction.  Not user-callable.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (ghb)
    C = GhB (gbmex_cat (ghb, Tiles)) ;
else
    C = GrB (gbmex_cat (ghb, Tiles)) ;
end

