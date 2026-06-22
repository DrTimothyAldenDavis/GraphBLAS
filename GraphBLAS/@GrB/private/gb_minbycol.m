function C = gb_minbycol (op, A)
%GB_MINBYCOL min, by column
% Implements C = min (A, [ ], 1)

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

ghb = 1 ;     % 0 for GrB, 1 for GhB

% C = min (A, [ ], 1) reduces each col to a scalar; C is 1-by-n
desc.in0 = 'transpose' ;
C = GrB (gbmex_vreduce (ghb, op, A, desc)) ;

% if C(j) > 0, but if A(:,j) is sparse, then assign C(j) = 0.
ctype = gbmex_type (C) ;

    % d (j) = number of entries in A(:,j); d (j) not present if A(:,j) empty
    [m, n] = gbmex_size (A) ;
    d = GrB (gbmex_degree (ghb, A, 'col')) ;
    % d (j) is an explicit zero if A(:,j) has 1 to m-1 entries
    s = GrB (gbmex_select (ghb, d, '<', int64 (m))) ;
    zero = GrB (0, ctype) ;
    if (gbmex_nvals (s) == n)
        % all columns A(:,j) have between 1 and m-1 entries
        C = GrB (gbmex_apply2 (ghb, op, C, zero)) ;
    else
        z = GrB (gbmex_apply2 (ghb, ['2nd.' ctype], s, zero)) ;
        % if z (j) is between 1 and m-1 and C (j) > 0 then C (j) = 0
        C = GrB (gbmex_eadd (ghb, op, C, z)) ;
    end

C = GrB (gbmex_trans (ghb, C)) ;

