function s = gb_isa (ghb, G, type)
%GB_ISA implements GrB/isa and GhB/isa.  Not user-callable.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (isequal (type, 'GrB') || (ghb && isequal (type, 'GhB')))
    % GraphBLAS matrics have a class name of 'GrB' or 'GhB'.
    % For GrB matrices, isa (G, 'GrB') is true but isa (G, 'GhB') is false.
    % For GhB matrices, both isa (G, 'GrB') and isa (G, 'GhB') are true.
    s = true ;
elseif isequal (type, 'numeric')
    % all GraphBLAS matrices are numeric
    s = true ;
elseif (isequal (type, 'float'))
    % GraphBLAS double, single, and complex matrices are 'float'
    s = isfloat (G) ;
elseif (isequal (type, 'integer'))
    % GraphBLAS int* and uint* matrices are 'integer'
    s = isinteger (G) ;
elseif (isequal (gbmex_type (G), type))
    % specific cases, such as isa (G, 'double'), isa (G, 'int8'), etc
    s = true ;
else
    % catch-all for cases not handled above
    s = builtin ('isa', G, type) ;
end

