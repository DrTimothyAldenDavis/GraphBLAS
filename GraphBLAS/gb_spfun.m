function C = gb_spfun (ghb, fun, G)
%GB_SPFUN implements GrB/spfun and GhB/spfun.  Not user-callable.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (gb_is_grb (G))
    G = struct (G) ;
end

if (ischar (fun))
    try
        C = gzb_apply (ghb, fun, G) ;
        return ;
    catch me %#ok<NASGU>
        % gzb_apply failed; fall through to feval below
    end
end

% 'fun' is not a string, or not a built-in GraphBLAS operator
[m, n] = gbmex_size (G) ;
desc.base = 'zero-based' ;
gbmex_wait (G) ;
[i, j, x] = gbmex_extracttuples (ghb, G, desc) ; % OK: zero-based integers
x = feval (fun, x) ;
C = gzb_build (ghb, i, j, x, m, n, '1st', desc) ;

