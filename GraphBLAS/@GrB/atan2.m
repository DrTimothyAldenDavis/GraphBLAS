function C = atan2 (A, B)
%ATAN2 four quadrant inverse tangent.
% C = atan2 (X,Y) is the 4 quadrant arctangent of the entries in X and Y.
%
% See also GrB/tan, GrB/tanh, GrB/atan, GrB/atanh.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

atype = gbtype (A) ;
btype = gbtype (B) ;

if (gb_contains (atype, 'complex') || gb_contains (btype, 'complex'))
    error ('GrB:error', 'inputs must be real') ;
end

% cast A and/or B to double, if not already a floating-point type
if (gb_isfloat (atype))
    if (gb_isfloat (btype))
        C = gb_atan2 (A, B) ;
    else
        b = gbnew (B, 'double') ;
        C = gb_atan2 (A, b) ;
        gbdelete (b) ;
    end
else
    a = gbnew (A, 'double') ;
    if (gb_isfloat (btype))
        C = gb_atan2 (a, B) ;
    else
        b = gbnew (B, 'double') ;
        C = gb_atan2 (a, b) ;
        gbdelete (b) ;
    end
    gbdelete (a) ;
end

