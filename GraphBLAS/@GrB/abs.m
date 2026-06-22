function C = abs (G)
%ABS absolute value.
% C = abs (G) is the absolute value of each entry of G.  C is always real,
% even if C is complex.
%
% See also GrB/sign.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

ghb = 1 ;     % 0 for GrB, 1 for GhB

if (gb_issigned (gbmex_type (G)))
    C = GrB (gbmex_apply (ghb, 'abs', G)) ;
else
    C = GrB (G) ;
end

