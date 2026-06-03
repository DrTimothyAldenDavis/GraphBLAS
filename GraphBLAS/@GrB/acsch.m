function C = acsch (G)
%ACSCH inverse hyperbolic cosecant.
% C = acsch (G) is the inverse hyberbolic cosecant of each entry G.  Since
% acsch (0) is nonzero, C is a full matrix.
%
% See also GrB/csc, GrB/acsc, GrB/csch.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

type = gbtype (G) ;
if (~gb_isfloat (type))
    type = 'double' ;
end

S = gbfull (G, type) ;
T = gbapply ('minv', S) ;
gbdelete (S) ;
C = GrB (gbapply ('asinh', T)) ;
gbdelete (T) ;

