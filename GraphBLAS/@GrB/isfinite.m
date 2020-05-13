function C = isfinite (G)
%ISFINITE True for finite elements.
% C = isfinite (G) returns a GraphBLAS logical matrix where C(i,j) = true
% if G(i,j) is finite.  C is a fully populated matrix.
%
% See also GrB/isnan, GrB/isinf.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights
% Reserved. http://suitesparse.com.  See GraphBLAS/Doc/License.txt.

[m, n] = size (G) ;
if (isfloat (G) && m > 0 && n > 0)
    C = GrB.apply ('isfinite', full (G)) ;
else
    % C is all true
    C = GrB (true (m, n)) ;
end

