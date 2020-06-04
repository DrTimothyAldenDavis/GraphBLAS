function C = isinf (G)
%ISINF true for infinite elements.
% C = isinf (G) returns a GraphBLAS logical matrix where C(i,j) = true
% if G(i,j) is infinite.
%
% See also GrB/isnan, GrB/isfinite.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights
% Reserved. http://suitesparse.com.  See GraphBLAS/Doc/License.txt.

G = G.opaque ;

if (gb_isfloat (gbtype (G)) && gbnvals (G) > 0)
    C = GrB (gbapply ('isinf', G)) ;
else
    % C is all false
    [m, n] = gbsize (G) ;
    C = GrB (m, n, 'logical') ;
end

