function C = isnan (G)
%ISNAN True for NaN elements.
% C = isnan (G) for a GraphBLAS matrix G returns a GraphBLAS logical
% matrix with C(i,j)=true if G(i,j) is NaN.
%
% See also GrB/isinf, GrB/isfinite.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights
% Reserved. http://suitesparse.com.  See GraphBLAS/Doc/License.txt.

if (isfloat (G) && GrB.entries (G) > 0)
    C = GrB.apply ('isnan', G) ;
else
    % C is all false
    [m, n] = size (G) ;
    C = GrB (m, n, 'logical') ;
end

