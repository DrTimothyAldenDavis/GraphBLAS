function C = repmat (G, m, n)
%REPMAT Replicate and tile a GraphBLAS matrix.
% C = repmat (G, m, n)  % constructs an m-by-n tiling of the gb matrix A
% C = repmat (G, [m n]) % same as C = repmat (A, m, n)
% C = repmat (G, n)     % constructs an n-by-n tiling of the gb matrix G
%
% See also kron, gb.gbkron.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

if (nargin == 3)
    R = ones (m, n, 'logical') ;
else
    R = ones (m, 'logical') ;
end
C = gb.gbkron (['2nd.' gb.type(G)], R, G) ;

