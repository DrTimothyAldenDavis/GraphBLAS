function X = gb_nonzeros (ghb, G)
%GB_NONZEROS implements GrB/nonzeros and GhB/nonzeros.  Not user-callable.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

S = gzb_select (ghb, 'nonzero', G) ;
gbmex_wait (S) ;
X = gbmex_extractvalues (S) ;

