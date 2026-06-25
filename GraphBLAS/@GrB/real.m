function C = real (G)
%REAL complex real part.
% C = real (G) returns the real part of G.
%
% See also GrB/conj, GrB/imag.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

ghb = 0 ;     % 0 for GrB, 1 for GhB

if (gb_contains (gbmex_type (G), 'complex'))
    C = GrB (gbmex_apply (ghb, 'creal', G)) ;
else
    % G is already real
    C = GrB (G) ;
end

