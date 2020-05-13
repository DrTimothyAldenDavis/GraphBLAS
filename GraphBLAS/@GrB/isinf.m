function C = isinf (G)
%ISINF true for infinite elements.
% C = isinf (G) returns a GraphBLAS logical matrix where C(i,j) = true
% if G(i,j) is infinite.
%
% See also GrB/isnan, GrB/isfinite.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights
% Reserved. http://suitesparse.com.  See GraphBLAS/Doc/License.txt.

if (isfloat (G) && GrB.entries (G) > 0)
    C = GrB.apply ('isinf', G) ;
else
    % C is all false
    [m, n] = size (G) ;
    C = GrB (m, n, 'logical') ;
end

