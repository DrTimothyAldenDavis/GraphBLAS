function C = full (X, type, identity)
%FULL convert a matrix into a GraphBLAS dense matrix.
% C = full (X, type, identity) converts the matrix X into a GraphBLAS
% dense matrix C of the given type, by inserting identity values.  The
% type may be any GraphBLAS type: 'double', 'single', 'logical', 'int8'
% 'int16' 'int32' 'int64' 'uint8' 'uint16' 'uint32' 'uint64', or in the
% future, 'complex'.  If not present, the type defaults to the same type
% as G, and the identity defaults to zero.  X may be any matrix
% (GraphBLAS, MATLAB sparse or full).  To use this method for a MATLAB
% matrix A, use a GraphBLAS identity value such as gb(0), or use C = full
% (gb (A)).  Note that issparse (C) is true, since issparse (G) is true
% for any GraphBLAS matrix G.
%
% Examples:
%
%   G = gb (sprand (5, 5, 0.5))         % GraphBLAS sparse matrix
%   C = full (G)                        % add explicit zeros
%   C = full (G, 'double', inf)         % add explicit inf's
%
%   A = speye (2)  
%   C = full (gb (A), 'double', 0)      % full gb matrix C, from A
%   C = full (gb (A))                   % same matrix C
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

C = gb (gbfull (X, type, identity, struct ('kind', 'gb'))) ;

