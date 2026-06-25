function C = fix (G)
%FIX Round towards zero.
% C = fix (G) rounds the entries in the matrix G to the nearest integers
% towards zero.
%
% See also GrB/ceil, GrB/floor, GrB/round.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

ghb = 0 ;     % 0 for GrB, 1 for GhB

if (gb_isfloat (gbmex_type (G)))
    C = gzb_apply (ghb, 'trunc', G) ;
else
    C = gzb (ghb, G) ;
end

