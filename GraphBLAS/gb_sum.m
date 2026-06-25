function C = gb_sum (op, G, option)
%GB_SUM C = sum (G) or C = any (G)

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

ghb = 0 ;     % 0 for GrB, 1 for GhB

if (nargin == 2)
    % C = sum (G)
    if (gb_isvector (G))
        option = 'all' ;
    else
        option = 1 ;
    end
end

switch (option)

    case { 'all' }

        % C = sum (G, 'all'), reducing all entries to a scalar
        C = gzb_reduce (ghb, op, G) ;

    case { 1 }

        % C = sum (G, 1) reduces each column to a scalar,
        % giving a 1-by-n row vector.
        desc.in0 = 'transpose' ;
        T = gzb_vreduce (ghb, op, G, desc) ;
        C = gzb_trans (ghb, T) ;

    case { 2 }

        % C = sum (G, 2) reduces each row to a scalar,
        % giving an m-by-1 column vector.
        C = gzb_vreduce (ghb, op, G) ;

    otherwise

        error ('GrB:error', 'unknown option') ;
end

