function s = gb_issymmetric (G_arg, option, herm)
%GB_ISSYMMETRIC check if symmetric or Hermitian
% Implements issymmetric (G,option) and ishermitian (G,option).

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

% FUTURE: this can be much faster; see spsym in CHOLMOD.

[m, n, type] = gbmex_size (G_arg) ;

if (m ~= n)

    s = false ;

else

    if (isequal (type, 'logical'))
        G = GrB (G_arg, 'double') ;
    else
        G = G_arg ;
    end

    if (herm && gb_contains (type, 'complex'))
        % T = G', complex conjugate transpose
        desc.in0 = 'transpose' ;
        T = GrB (gbmex_apply ('conj', G, desc)) ;
    else
        % T = G.', array transpose
        T = GrB (gbmex_trans (G)) ;
    end

    switch (option)

        case { 'skew' }

            % G is skew symmetric/Hermitian if G+T is zero
            s = (gbmex_norm (gb_eadd (G, '+', T), 1) == 0) ;

        case { 'nonskew' }

            % G is symmetric/Hermitian if G-T is zero
            s = (gbmex_normdiff (G, T, 1) == 0) ;

        otherwise

            error ('GrB:error', 'invalid option') ;

    end

    if (s)
        % also check the pattern; G might have explicit zeros
        S = gb_spones (G, 'logical') ;
        T = GrB (gbmex_trans (S)) ;
        s = gbmex_isequal (S, T) ;
    end
end

