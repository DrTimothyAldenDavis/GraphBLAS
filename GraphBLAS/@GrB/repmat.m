function C = repmat (G, m, n)
%REPMAT replicate and tile a matrix.
% C = repmat (G, m, n)  % constructs an m-by-n tiling of the matrix G
% C = repmat (G, [m n]) % same as C = repmat (A, m, n)
% C = repmat (G, n)     % constructs an n-by-n tiling of the matrix G
%
% See also GrB/kron, GrB.kronecker.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
% SPDX-License-Identifier: Apache-2.0

ghb = 0 ;     % 0 for GrB, 1 for GhB

if (nargin < 3)
    C = gb_repmat (ghb, G, m) ;
else
    C = gb_repmat (ghb, G, m, n) ;
end

