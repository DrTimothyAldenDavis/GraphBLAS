function C = gb_minbycol (op, A)
%GB_MINBYCOL min, by column
% Implements C = min (A, [ ], 1)

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

% C = min (A, [ ], 1) reduces each col to a scalar; C is 1-by-n
desc.in0 = 'transpose' ;
c = GrB (gbvreduce (op, A, desc)) ;

% if c(j) > 0, but if A(:,j) is sparse, then assign c(j) = 0.
ctype = gbtype (c) ;

    % d (j) = number of entries in A(:,j); d (j) not present if A(:,j) empty
    [m, n] = gbsize (A) ;
    d = GrB (gbdegree (A, 'col')) ;
    % d (j) is an explicit zero if A(:,j) has 1 to m-1 entries
    s = GrB (gbselect (d, '<', int64 (m))) ;

    zero = GrB (gbnew (0, ctype)) ;
    if (gbnvals (s) == n)
        % all columns A(:,j) have between 1 and m-1 entries
        T = GrB (gbapply2 (op, c, zero)) ;
    else
        z = GrB (gbapply2 (['2nd.' ctype], s, zero)) ;
        % if z (j) is between 1 and m-1 and C (j) > 0 then C (j) = 0
        T = GrB (gbeadd (op, c, z)) ;
    end

C = GrB (gbtrans (T)) ;

