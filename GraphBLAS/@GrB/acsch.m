function C = acsch (G)
%ACSCH inverse hyperbolic cosecant.
% C = acsch (G) computes the inverse hyberbolic cosecant of each entry of a
% GraphBLAS matrix G.  Since acsch (0) is nonzero, C is a full matrix.
%
% See also GrB/csc, GrB/acsc, GrB/csch.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights
% Reserved. http://suitesparse.com.  See GraphBLAS/Doc/License.txt.

G = G.opaque ;
type = gbtype (G) ;
if (~gb_isfloat (type))
    G = GrB (G, 'double') ;
end

C = GrB (gbapply ('asinh', gbapply ('minv', gbfull (G)))) ;

