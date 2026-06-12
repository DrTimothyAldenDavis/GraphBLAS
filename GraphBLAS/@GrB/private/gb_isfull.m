function s = gb_isfull (A)
%GB_ISFULL determine if all entries are present in a GraphBLAS struct.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

[m, n] = gbmex_size (A) ;
if (isinteger (m))
    % gbmex_size returms m and n as integer if either m or n are larger
    % than flintmax.  In this case, A must be sparse.
    s = false ;
else
    % FIXME: gbmex_nvals requires a wait
    s = (m*n == gbmex_nvals (A)) ;
end

