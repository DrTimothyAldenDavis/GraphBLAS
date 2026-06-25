function C = erf (G)
%ERF error function.
% C = erf (G) computes the error function of each entry of G.
% G must be real.
%
% See also GrB/erfc, erfcx, erfinv, erfcinv.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

ghb = 0 ;     % 0 for GrB, 1 for GhB

type = gbmex_type (G) ;
if (gb_contains (type, 'complex'))
    error ('GrB:error', 'input must be real') ;
end
if (~gb_isfloat (type))
    op = 'erf.double' ;
else
    op = 'erf' ;
end

C = GrB (gbmex_apply (ghb, op, G)) ;

