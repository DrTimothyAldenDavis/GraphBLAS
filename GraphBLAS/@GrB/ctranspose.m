function C = ctranspose (G)
%CTRANSPOSE C = G', transpose a GraphBLAS matrix.
% C = G' is the complex conjugate transpose of G.
%
% See also GrB.trans, GrB/transpose, GrB/conj.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

ghb = 0 ;     % 0 for GrB, 1 for GhB

if (gb_contains (gbmex_type (G), 'complex'))
    desc.in0 = 'transpose' ;
    C = gzb_apply (ghb, 'conj', G, desc) ;
else
    C = gzb_trans (ghb, G) ;
end

