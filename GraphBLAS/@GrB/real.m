function C = real (G)
%REAL complex real part.
% C = real (G) returns the real part of G.
%
% See also GrB/conj, GrB/imag.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (gb_contains (gbmex_type (G), 'complex'))
    C = GrB (gbmex_apply ('creal', G)) ;
else
    % G is already real
    C = GrB (G) ;
end

