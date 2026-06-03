function C = gb_scalar_to_full (m, n, type, fmt, scalar)
%GB_SCALAR_TO_FULL expand a scalar into a full matrix

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (~isempty (strfind (fmt, 'by row'))) %#ok<STREMP>
    fmt = 'by row' ;
else
    fmt = 'by col' ;
end

E = GrB (m, n, type, fmt) ;
S = GrB (gbfull (scalar)) ;
C = GrB (gbsubassign (E, S)) ;

