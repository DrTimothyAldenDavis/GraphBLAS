function C = hypot (A_arg, B_arg)
%HYPOT robust computation of the square root of sum of squares.
% C = hypot (A,B) computes sqrt (abs (A).^2 + abs (B).^2) accurately.
% If A and B are matrices, the pattern of C is the set union of A and B.
% If one of A or B is a nonzero scalar, the scalar is expanded into a
% full matrix the size of the other matrix, and the result is a full
% matrix.
%
% See also GrB/abs, GrB/norm, GrB/sqrt, GrB/plus, GrB.eadd.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

ghb = 0 ;     % 0 for GrB, 1 for GhB

atype = gbmex_type (A_arg) ;
btype = gbmex_type (B_arg) ;

if (gb_contains (atype, 'complex'))
    A = gzb_apply (ghb, 'abs', A_arg) ;
elseif (~gb_isfloat (atype))
    A = gzb (ghb, A_arg, 'double') ;
else
    % use A_arg as-is
    A = A_arg ;
end

if (gb_contains (btype, 'complex'))
    B = gzb_apply (ghb, 'abs', B_arg) ;
elseif (~gb_isfloat (btype))
    B = gzb (ghb, B_arg, 'double') ;
else
    % use B_arg as-is
    B = B_arg ;
end

C = gzb_apply (ghb, 'abs', gb_eadd (A, 'hypot', B)) ;

