function C = csc (G)
%CSC cosecant.
% C = csc (G) computes the cosecant of each entry of a GraphBLAS matrix G.
% Since csc (0) is nonzero, C is a full matrix.
%
% See also GrB/acsc, GrB/csch, GrB/acsch.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights
% Reserved. http://suitesparse.com.  See GraphBLAS/Doc/License.txt.

G = G.opaque ;

if (~gb_isfloat (gbtype (G)))
    G = gbnew (G, 'double') ;
end

C = GrB (gbapply ('minv', gbfull (gbapply ('sin', G)))) ;

