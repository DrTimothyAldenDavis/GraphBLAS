function C = round (G)
%ROUND round entries of a matrix to the nearest integers.
% C = round (G) rounds the entries of G to the nearest integers.
%
% Note: the additional parameters of the built-in round function,
% round(x,n) and round (x,n,type), are not supported.
%
% See also GrB/ceil, GrB/floor, GrB/fix.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

ghb = 1 ;     % 0 for GrB, 1 for GhB

% FUTURE: round (x,n) and round (x,n,type)

if (gb_isfloat (gbmex_type (G)))
    C = GrB (gbmex_apply (ghb, 'round', G)) ;
else
    C = GrB (G) ;
end

