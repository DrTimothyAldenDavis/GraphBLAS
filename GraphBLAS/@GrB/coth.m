function C = coth (G)
%COTH hyperbolic cotangent.
% C = coth (G) computes the hyperbolic cotangent of each entry of a
% GraphBLAS matrix G.  Since coth (0) is nonzero, C is a full matrix.
%
% See also GrB/cot, GrB/acot, GrB/acoth.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights
% Reserved. http://suitesparse.com.  See GraphBLAS/Doc/License.txt.

G = G.opaque ;

if (~gb_isfloat (gbtype (G)))
    G = gbnew (G, 'double') ;
end

C = GrB (gbapply ('minv', gbfull (gbapply ('tanh', G)))) ;

