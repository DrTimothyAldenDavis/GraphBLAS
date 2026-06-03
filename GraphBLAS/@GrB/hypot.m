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

atype = gbtype (A_arg) ;
btype = gbtype (B_arg) ;

if (gb_contains (atype, 'complex'))
    A = GrB (gbapply ('abs', A_arg)) ;
elseif (~gb_isfloat (atype))
    A = GrB (A_arg, 'double') ;
else
    % use A_arg as-is
    A = A_arg ;
end

if (gb_contains (btype, 'complex'))
    B = GrB (gbapply ('abs', B_arg)) ;
elseif (~gb_isfloat (btype))
    B = GrB (B_arg, 'double') ;
else
    % use B_arg as-is
    B = B_arg ;
end

C = GrB (gbapply ('abs', gb_eadd (A, 'hypot', B))) ;

