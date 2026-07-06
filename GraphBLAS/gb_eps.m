function C = gb_eps (ghb, G)
%GB_EPS implements GrB/eps and GhB/eps.  Not user-callable.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

% FUTURE: GraphBLAS should have a built-in eps unary operator.

% FUTURE: there should be a sparse version of 'eps'.
% C is full because eps (0) is 2^(-1024).

% convert to a built-in full matrix and use the built-in eps
switch (gbmex_type (G))

    case { 'single' }
        T = eps (gb_single (ghb, gb_full (ghb, G))) ;

    case { 'double' }
        T = eps (gb_double (ghb, gb_full (ghb, G))) ;

    case { 'single complex' }
        T = max (eps (gb_single (ghb, gb_real (ghb, G))), ...
                 eps (gb_single (ghb, gb_imag (ghb, G)))) ;

    case { 'double complex' }
        T = max (eps (gb_double (ghb, gb_real (ghb, G))), ...
                 eps (gb_double (ghb, gb_imag (ghb, G)))) ;

    otherwise
        error ('GrB:error', 'input must be floating-point') ;

end

C = gzb (ghb, T) ;

