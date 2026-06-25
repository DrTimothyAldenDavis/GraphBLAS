function C = gb_maxbycol (op, A)
%GB_MAXBYCOL max, by column
% Implements C = max (A, [ ], 1)

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

ghb = 0 ;     % 0 for GrB, 1 for GhB

% C = max (A, [ ], 1) reduces each col to a scalar; C is 1-by-n
desc.in0 = 'transpose' ;
C = gzb_vreduce (ghb, op, A, desc) ;

% if C(j) < 0, but if A(:,j) is sparse, then assign C(j) = 0.
ctype = gbmex_type (C) ;

if (gb_issigned (ctype))
    % d (j) = number of entries in A(:,j); d (j) not present if A(:,j) empty
    [m, n] = gbmex_size (A) ;
    d = gzb_degree (ghb, A, 'col') ;
    % s (j) is an explicit zero if A(:,j) has 1 to m-1 entries
    s = gzb_select (ghb, d, '<', int64 (m)) ;
    zero = gzb (ghb, 0, ctype) ;
    if (gbmex_nvals (s) == n)
        % all columns A(:,j) have between 1 and m-1 entries
        C = gzb_apply2 (ghb, op, C, zero) ;
    else
        z = gzb_apply2 (ghb, ['2nd.' ctype], s, zero) ;
        % if z (j) is between 1 and m-1 and C (j) < 0 then C (j) = 0
        C = gzb_eadd (ghb, op, C, z) ;
    end
end

C = gzb_trans (ghb, C) ;

