function C = csch (G)
%CSCH hyperbolic cosecant.
% C = csch (G) is the hyperbolic cosecant of each entry of G.  Since
% csch(0) is nonzero, C is a full matrix.
%
% See also GrB/csc, GrB/acsc, GrB/acsch.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

if (~gb_isfloat (gbtype (G)))
    op = 'sinh.double' ;
else
    op = 'sinh' ;
end

C = GrB (gbapply ('minv', GrB (gbfull (GrB (gbapply (op, G)))))) ;

