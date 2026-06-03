function C = gb_maxbyrow (op, A)
%GB_MAXBYROW max, by row
% Implements C = max (A, [ ], 2)

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

% C = max (A, [ ], 2) reduces each row to a scalar; C is m-by-1
c = GrB (gbvreduce (op, A)) ;

% if c(i) < 0, but if A(i,:) is sparse, then assign c(i) = 0.
ctype = gbtype (c) ;

if (gb_issigned (ctype))
    % d (i) = number of entries in A(i,:); d (i) not present if A(i,:) empty
    [m, n] = gbsize (A) ;
    d = GrB (gbdegree (A, 'row')) ;
    % d (i) is an explicit zero if A(i,:) has 1 to n-1 entries
    s = GrB (gbselect (d, '<', int64 (n))) ;
    zero = GrB (0, ctype) ;
    if (gbnvals (s) == m)
        % all rows A(i,:) have between 1 and n-1 entries
        C = GrB (gbapply2 (op, c, zero)) ;
    else
        z = GrB (gbapply2 (['2nd.' ctype], s, zero)) ;
        % if z(i) is between 1 and n-1 and c(i) < 0 then C(i) = 0
        C = GrB (gbeadd (op, c, z)) ;
    end
else
    C = c ;
end

