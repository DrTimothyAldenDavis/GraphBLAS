function C = isfinite (G)
%ISFINITE True for finite elements.
% See also isnan, isinf.

% FUTURE: this could be much faster as a mexFunction.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

[m n] = size (G) ;
if (isfloat (G) && m > 0 && n > 0)
    [i j x] = gb.extracttuples (full (G), struct ('kind', 'zero-based')) ;
    C = gb.build (i, j, isfinite (x), m, n) ;
else
    % C is all true
    C = gb (true (m, n)) ;
end

