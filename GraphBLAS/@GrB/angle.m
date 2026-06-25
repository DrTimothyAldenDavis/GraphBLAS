function C = angle (G)
%ANGLE phase angle.
% C = angle (G) is the phase angle of each entry of G.
%
% See also GrB/abs.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

ghb = 0 ;     % 0 for GrB, 1 for GhB

[m, n, type] = gbmex_size (G) ;
if (gb_contains (type, 'complex'))
    C = gzb_apply (ghb, 'carg', G) ;
else
    % C is all zero
    C = gzb (ghb, m, n, type) ;
end

