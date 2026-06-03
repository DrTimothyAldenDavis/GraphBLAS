function C = gb_prod (op, type, G, option)
%GB_PROD C = prod (G), using the given operator and type
% Implements C = prod (G) and C = all (G).

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

[m, n] = gbsize (G) ;

if (nargin == 3)
    % C = prod (G)
    if (m == 1 || n == 1)
        option = 'all' ;
    else
        option = 1 ;
    end
end


switch (option)

    case { 'all' }

        % C = prod (G, 'all'), reducing all entries to a scalar
        if (m*n == gbnvals (G))
            C = gbreduce (op, G) ;
        else
            C = gbnew (0, type) ;
        end

    case { 1 }

        % C = prod (G,1) reduces each column to a scalar,
        % giving a 1-by-n row vector.
        % M = find (column degree of G == m)
        M = gbselect (gbdegree (G, 'col'), '==', int64 (m)) ;
        Cin = gbnew (n, 1, type) ;
        % C<M> = op (G')
        desc.in0 = 'transpose' ;
        T = gbvreduce (Cin, M, op, G, desc) ;
        C = gbtrans (T) ;
        gbdelete (T) ;
        gbdelete (Cin) ;
        gbdelete (M) ;

    case { 2 }

        % C = prod (G,2) reduces each row to a scalar,
        % giving an m-by-1 column vector.
        % M = find (row degree of G == n)
        d = gbdegree (G, 'row') ;
        M = gbselect (d, '==', int64 (n)) ;
        gbdelete (d) ;
        % C<M> = op (G)
        Cin = gbnew (m, 1, type) ;
        C = gbvreduce (Cin, M, op, G) ;
        gbdelete (Cin) ;
        gbdelete (M) ;

    otherwise

        error ('GrB:error', 'unknown option') ;
end

