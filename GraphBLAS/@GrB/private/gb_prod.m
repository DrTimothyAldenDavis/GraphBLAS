function C = gb_prod (op, type, G, option)
%GB_PROD C = prod (G), using the given operator and type
% Implements C = prod (G) and C = all (G).

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

ghb = 1 ;     % 0 for GrB, 1 for GhB

[m, n] = gbmex_size (G) ;

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
        if (gb_isfull (G))
            C = GrB (gbmex_reduce (ghb, op, G)) ;
        else
            C = GrB (0, type) ;
        end

    case { 1 }

        % C = prod (G,1) reduces each column to a scalar,
        % giving a 1-by-n row vector.
        % M = find (column degree of G == m)
        M = GrB (gbmex_select (ghb, GrB (gbmex_degree (ghb, G, 'col')), '==', int64 (m))) ;
        Cin = GrB (n, 1, type) ;
        % C<M> = op (G')
        desc.in0 = 'transpose' ;
        C = GrB (gbmex_trans (ghb, GrB (gbmex_vreduce (ghb, Cin, M, op, G, desc)))) ;

    case { 2 }

        % C = prod (G,2) reduces each row to a scalar,
        % giving an m-by-1 column vector.
        % M = find (row degree of G == n)
        M = GrB (gbmex_select (ghb, GrB (gbmex_degree (ghb, G, 'row')), '==', int64 (n))) ;
        % C<M> = op (G)
        Cin = GrB (m, 1, type) ;
        C = GrB (gbmex_vreduce (ghb, Cin, M, op, G)) ;

    otherwise

        error ('GrB:error', 'unknown option') ;
end

