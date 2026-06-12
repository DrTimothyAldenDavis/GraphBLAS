function C = gamma (G)
%GAMMA gamma function.
% C = gamma (G) is the gamma function of each entry of G.
% Since gamma (0) = inf, the result is a full matrix.  G must be real.
%
% See also GrB/gammaln.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

type = gbmex_type (G) ;
if (gb_contains (type, 'complex'))
    error ('GrB:error', 'input must be real') ;
end
if (~gb_isfloat (type))
    type = 'double' ;
end

C = GrB (gbmex_apply ('gamma', GrB (gbmex_full (G, type)))) ;

