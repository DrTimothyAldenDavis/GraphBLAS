function C = sin (G)
%SIN sine.
% C = sin (G) computes the sine of each entry of a GraphBLAS matrix G.
%
% See also GrB/asin, GrB/sinh, GrB/asinh.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights
% Reserved. http://suitesparse.com.  See GraphBLAS/Doc/License.txt.

G = G.opaque ;

if (~gb_isfloat (gbtype (G)))
    G = gbnew (G, 'double') ;
end

C = GrB (gbapply ('sin', G)) ;

