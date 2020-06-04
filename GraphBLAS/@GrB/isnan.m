function C = isnan (G)
%ISNAN True for NaN elements.
% C = isnan (G) for a GraphBLAS matrix G returns a GraphBLAS logical
% matrix with C(i,j)=true if G(i,j) is NaN.
%
% See also GrB/isinf, GrB/isfinite.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights
% Reserved. http://suitesparse.com.  See GraphBLAS/Doc/License.txt.

G = G.opaque ;

if (gb_isfloat (gbtype (G)) && gbnvals (G) > 0)
    C = GrB (gbapply ('isnan', G)) ;
else
    % C is all false
    [m, n] = gbsize (G) ;
    C = GrB (m, n, 'logical') ;
end

