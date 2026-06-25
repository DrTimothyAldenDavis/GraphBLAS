function C = csc (G)
%CSC cosecant.
% C = csc (G) is the cosecant of each entry of G.  Since csc (0) is
% nonzero, C is a full matrix.
%
% See also GrB/acsc, GrB/csch, GrB/acsch.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

ghb = 0 ;     % 0 for GrB, 1 for GhB

if (~gb_isfloat (gbmex_type (G)))
    op = 'sin.double' ;
else
    op = 'sin' ;
end

C = GrB (gbmex_apply (ghb, 'minv', GrB (gbmex_full (ghb, GrB (gbmex_apply (ghb, op, G)))))) ;

