function C = gb_abs (ghb, G)
%GB_ABS implements GrB/abs and GhB/abs.  Not user-callable.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (gb_issigned (gbmex_type (G)))
    C = gzb_apply (ghb, 'abs', G) ;
else
    C = gzb (ghb, G) ;
end

