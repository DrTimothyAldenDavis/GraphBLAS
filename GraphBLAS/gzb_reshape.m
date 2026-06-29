function C = gzb_reshape (ghb, A, mnew, nnew, by_col)
%GZB_RESHAPE: wrapper for gbmex_reshape mexFunction.  Not user-callable.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (ghb)
    C = GhB (gbmex_reshape (ghb, A, mnew, nnew, by_col)) ;
else
    C = GrB (gbmex_reshape (ghb, A, mnew, nnew, by_col)) ;
end

