function C = gb_maxbyrow (op, A)
%GB_MAXBYROW max, by row
% Implements C = max (A, [ ], 2)

% C = max (A, [ ], 2) reduces each row to a scalar; C is m-by-1
C = gbvreduce (op, A) ;

% if C(i) < 0, but the A(i,:) is sparse, then assign C(i) = 0.
ctype = gbtype (C) ;

if (gb_issigned (ctype))
    % d (i) = true if A (i,:) has fewer than n entries
    [~, n] = gbsize (A) ;
    d = gbdegree (A, 'row') ;
    d = gbemult (d, '<', gb_expand (n, d)) ;
    % c = (C < 0)
    c = gbemult (C, '<', gb_expand (0, C)) ;
    % mask = c & d
    mask = gbemult (c, '&', d) ;
    % delete entries in C where mask is true
    [m, n] = gbsize (mask) ;
    C = gbsubassign (C, mask, gbnew (m, n, ctype)) ;
end

