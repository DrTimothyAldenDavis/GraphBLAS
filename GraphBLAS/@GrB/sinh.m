function C = sinh (G)
%SINH hyperbolic sine.
% C = sinh (G) computes the hyperbolic sine of each entry of a GraphBLAS
% matrix G.
%
% See also GrB/sin, GrB/asin, GrB/asinh.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights
% Reserved. http://suitesparse.com.  See GraphBLAS/Doc/License.txt.

G = G.opaque ;

if (~gbisfloat (gbtype (G)))
    G = gbnew (G, 'double') ;
end

C = GrB (gbapply ('sinh', G)) ;

