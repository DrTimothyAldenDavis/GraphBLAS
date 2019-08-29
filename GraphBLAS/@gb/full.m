function F = full (X, type, identity)
%FULL convert a matrix into a GraphBLAS dense matrix.
% F = full (X, type, identity) converts the matrix X into a GraphBLAS dense
% matrix F of the given type, by inserting identity values.  The type may
% be any GraphBLAS type: 'double', 'single', 'logical', 'int8' 'int16'
% 'int32' 'int64' 'uint8' 'uint16' 'uint32' 'uint64', or in the future,
% 'complex'.  If not present, the type defaults to the same type as G, and
% the identity defaults to zero.  X may be any matrix (GraphBLAS, MATLAB
% sparse or full).  To use this method for a MATLAB matrix A, use a
% GraphBLAS identity value such as gb(0), or use F = full (gb (A)).  Note
% that issparse (F) is true, since issparse (G) is true for any GraphBLAS
% matrix G.
%
% Examples:
%
%   G = gb (sprand (5, 5, 0.5))         % GraphBLAS sparse matrix
%   F = full (G)                        % add explicit zeros
%   F = full (G, 'double', inf)         % add explicit inf's
%
%   A = speye (2) ;
%   F = full (A, 'double', 0) ;         % full gb matrix F, from A
%   F = full (gb (A)) ;                 % same matrix F
%
% See also issparse, sparse, cast, gb.type, gb.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

if (isa (X, 'gb'))
    X = X.opaque ;
end
if (nargin < 2)
    type = gbtype (X) ;
end
if (nargin < 3)
    identity = 0 ;
end
if (isa (identity, 'gb'))
    identity = identity.opaque ;
end

F = gb (gbfull (X, type, identity, struct ('kind', 'gb'))) ;

