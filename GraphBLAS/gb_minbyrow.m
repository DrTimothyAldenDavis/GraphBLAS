function C = gb_minbyrow (ghb, op, A)
%GB_MINBYROW min, by row.  Not user-callable.
% Implements C = min (A, [ ], 2)

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

% C = min (A, [ ], 2) reduces each row to a scalar; C is m-by-1
C = gzb_vreduce (ghb, op, A) ;

% if C(i) > 0, but if A(i,:) is sparse, then assign C(i) = 0.
ctype = gbmex_type (C) ;

    % d (i) = number of entries in A(i,:); d (i) not present if A(i,:) empty
    [m, n] = gbmex_size (A) ;
    d = gzb_degree (ghb, A, 'row') ;
    % d (i) is an explicit zero if A(i,:) has 1 to n-1 entries
    s = gzb_select (ghb, d, '<', int64 (n)) ;
    zero = gzb (ghb, 0, ctype) ;
    if (gbmex_nvals (s) == m)
        % all rows A(i,:) have between 1 and n-1 entries
        C = gzb_apply2 (ghb, op, C, zero) ;
    else
        z = gzb_apply2 (ghb, ['2nd.' ctype], s, zero) ;
        % if d(i) is between 1 and n-1 and C(i) > 0 then C(i) = 0
        C = gzb_eadd (ghb, op, C, z) ;
    end

