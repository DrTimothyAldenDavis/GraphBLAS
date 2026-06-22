function C = ctranspose (G)
%CTRANSPOSE C = G', transpose a GraphBLAS matrix.
% C = G' is the complex conjugate transpose of G.
%
% See also GrB.trans, GrB/transpose, GrB/conj.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

ghb = 1 ;     % 0 for GrB, 1 for GhB

if (gb_contains (gbmex_type (G), 'complex'))
    desc.in0 = 'transpose' ;
    C = GrB (gbmex_apply (ghb, 'conj', G, desc)) ;
else
    C = GrB (gbmex_trans (ghb, G)) ;
end

