function C = imag (G)
%IMAG complex imaginary part.
% C = imag (G) returns the imaginary part of G.
%
% See also GrB/conj, GrB/real.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

ghb = 1 ;     % 0 for GrB, 1 for GhB

[m, n, type] = gbmex_size (G) ;

if (gb_contains (type, 'complex'))
    % C = imag (G) where G is complex
    C = GrB (gbmex_apply (ghb, 'cimag', G)) ;
else
    % G is real, so C = zeros (m,n)
    C = GrB (m, n, type) ;
end

