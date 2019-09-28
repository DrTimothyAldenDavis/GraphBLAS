function C = isnan (G)
%ISNAN True for NaN elements.
% C = isnan (G) for a GraphBLAS matrix G returns a GraphBLAS logical
% matrix with C(i,j)=true if G(i,j) is NaN.
%
% See also isinf, isfinite.

% FUTURE: this will be much faster as a mexFunction.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

[m, n] = size (G) ;
if (isfloat (G) && GrB.entries (G) > 0)
    [i, j, x] = GrB.extracttuples (G, struct ('kind', 'zero-based')) ;
    C = GrB.build (i, j, isnan (x), m, n) ;
else
    % C is all false
    C = GrB (m, n, 'logical') ;
end

