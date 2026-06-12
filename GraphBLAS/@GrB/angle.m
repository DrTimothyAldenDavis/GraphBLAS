function C = angle (G)
%ANGLE phase angle.
% C = angle (G) is the phase angle of each entry of G.
%
% See also GrB/abs.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

[m, n, type] = gbmex_size (G) ;
if (gb_contains (type, 'complex'))
    C = GrB (gbmex_apply ('carg', G)) ;
else
    % C is all zero
    C = GrB (m, n, type) ;
end

