function L = laplacian (A, type, check)
%GB.LAPLACIAN Graph Laplacian matrix
% L = laplacian (A) is the graph Laplacian of the matrix A.  spones(A) must be
% symmetric with no diagonal entries. The diagonal of L is the degree of the
% nodes.  That is, L(j,j) = sum (spones (A (:,j))).  For off-diagonal entries,
% L(i,j) = L(j,i) = -1 if the edge (i,j) exists in A.
%
% The type of L defaults to double.  With a second argument, the type of L can
% be specified, as L = laplacian (A,type); type may be 'double', 'single',
% 'int8', 'int16', 'int32', or 'int64'.  Be aware that integer overflow may
% occur with the smaller integer types.
%
% To check the input matrix, use gb.laplacian (A, 'double', 'check') ;
%
% L is returned as symmetric matrix.
%
% Example:
%
%   A = bucky ;
%   L = gb.laplacian (A)
%
% See also graph/laplacian.

% TODO: tests

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

if (nargin < 2)
    type = 'double' ;
end
if (nargin < 3)
    check = false ;
else
    check = isequal (check, 'check') ;
end

if (~ (isequal (type, 'double') || isequal (type, 'single') || ...
       isequal (type (1:3), 'int')))
    % type must be 'double', 'single', 'int8', 'int16', 'int32', or 'int64'.
    error ('invalid type') ;
end

A = gb.apply (['1.' type], A) ;

if (check)
    if (~issymmetric (A))
        error ('A must be symmetric') ;
    end
    if (gb.nvals (diag (A)) > 0)
        error ('A must have no diagonal entries') ;
    end
end

if (gb.isbycol (A))
    D = gb.vreduce ('+', A, struct ('in0', 'transpose')) ;
else
    D = gb.vreduce ('+', A) ;
end

% construct the Laplacian
L = - gb.offdiag (A) + diag (D) ;

