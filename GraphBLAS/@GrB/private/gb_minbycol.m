function C = gb_minbycol (op, A)
%GB_MINBYCOL min, by column
% Implements C = min (A, [ ], 1)

% C = min (A, [ ], 1) reduces each col to a scalar; C is 1-by-n
desc.in0 = 'transpose' ;
C = gbvreduce (op, A, desc) ;

% if C(j) > 0, but the A(:,j) is sparse, then assign C(j) = 0.
ctype = gbtype (C) ;

% if (gb_issigned (ctype))
    % d (i) = true if A (:,j) has fewer than m entries
    [m, ~] = gbsize (A) ;
    d = gbdegree (A, 'col') ;
    d = gbemult (d, '<', gb_expand (m, d)) ;
    % c = (C > 0)
    c = gbemult (C, '>', gb_expand (0, C)) ;
    % mask = c & d
    mask = gbemult (c, '&', d) ;
    % delete entries in C where mask is true
    [m, n] = gbsize (mask) ;
    C = gbsubassign (C, mask, gbnew (m, n, ctype)) ;
% end

C = gbtrans (C) ;

