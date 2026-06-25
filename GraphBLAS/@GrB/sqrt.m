function C = sqrt (G)
%SQRT square root.
% C = sqrt (G) is the square root of the entries of G.
%
% See also GrB.apply, GrB/hypot.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

ghb = 0 ;     % 0 for GrB, 1 for GhB

C = gb_trig ('sqrt', G) ;

if (gb_make_real (C))
    C = GrB (gbmex_apply (ghb, 'creal', C)) ;
end

