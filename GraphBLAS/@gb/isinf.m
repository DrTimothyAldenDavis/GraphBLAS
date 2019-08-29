function C = isinf (G)
%ISINF True for infinite elements.
% See also isnan, isfinite.

% FUTURE: this could be much faster as a mexFunction.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

[m n] = size (G) ;
if (isfloat (G) && gb.nvals (G) > 0)
    [i j x] = gb.extracttuples (G, struct ('kind', 'zero-based')) ;
    C = gb.build (i, j, isinf (x), m, n) ;
else
    % C is all false
    C = gb (m, n, 'logical') ;
end

