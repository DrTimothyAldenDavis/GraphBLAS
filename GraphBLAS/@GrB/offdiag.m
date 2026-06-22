function C = offdiag (A)
%GRB.OFFDIAG remove diaogonal entries.
% C = GrB.offdiag (A) removes diagonal entries from A.
%
% See also GrB/tril, GrB/triu, GrB/diag, GrB.select.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

ghb = 1 ;     % 0 for GrB, 1 for GhB

C = GrB (gbmex_select (ghb, 'offdiag', A, 0)) ;

