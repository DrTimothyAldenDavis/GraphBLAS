function C = gb_spones (G, type)
%GB_SPONES return pattern of GraphBLAS matrix.
% Implements C = spones (G).

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

ghb = 0 ;     % 0 for GrB, 1 for GhB

if (nargin == 1)
    switch (gbmex_type (G))
        case { 'single complex' }
            op = '1.single' ;
        case { 'double complex' }
            op = '1.double' ;
        otherwise
            op = '1' ;
    end
else
    if (~ischar (type))
        error ('GrB:error', 'type must be a string') ;
    end
    op = ['1.' type] ;
end

C = GrB (gbmex_apply (ghb, op, G)) ;

