function C = subsref (A, S)
%SUBSREF C = A(I,J) or C = A(I); extract submatrix of a GraphBLAS matrix.
% C = A(I,J) extracts the A(I,J) submatrix of the GraphBLAS matrix A.  With
% a single index, C = A(I) extracts a subvector C of a vector A.  Linear
% indexing of a matrix is not yet supported.
%
% x = A (M) for a logical matrix M constructs an nnz(M)-by-1 vector x, for
% MATLAB-style logical indexing.  A or M may be MATLAB sparse or full
% matrices, or GraphBLAS matrices, in any combination.  M must be either a
% MATLAB logical matrix (sparse or dense), or a GraphBLAS logical matrix;
% that is, gb.type (M) must be 'logical'.
%
% NOTE: GraphBLAS can construct huge sparse matrices, but they cannot
% always be indexed with A(I,J), because of a limitation of the colon
% notation in the MATLAB subsref method.  A colon expression is expanded
% into an explicit vector, but can be too big.  Use gb.extract in this
% case, which can be passed the three integers start:inc:fini.
%
% Example:
%
%   n = 1e14 ;
%   H = gb (n, n)               % a huge empty matrix
%   I = [1 1e9 1e12 1e14] ;
%   M = magic (4)
%   H (I,I) = M
%   J = {1, 1e13} ;             % represents 1:1e13 colon notation
%   C = gb.extract (H, J, J)    % this is very fast
%   E = H (1:1e13, 1:1e13)      % but this is not possible 
%
% See also subsasgn, gb.subassign, gb.assign, gb.extract.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

if (length (S) > 1)
    error ('nested indexing not supported') ;
end

if (~isequal (S.type, '()'))
    error ('index type %s not supported', S.type) ;
end

ndims = length (S.subs) ;

if (ndims == 1)
    if (isequal (gb.type (S.subs {1}), 'logical'))
        % C = A (M) for a logical indexing
        M = S.subs {1} ;
        if (isa (M, 'gb'))
            M = M.opaque ;
        end
        if (isa (A, 'gb'))
            A = A.opaque ;
        end
        C = gb (gblogextract (A, M)) ;
    else
        % C = A (I) for a vector A
        if (~isvector (A))
            error ('Linear indexing of a gb matrix is not yet supported') ;
        end
        [I, whole_vector] = get_index (S.subs (1)) ;
        if (size (A, 1) > 1)
            C = gb.extract (A, I, { }) ;
        else
            C = gb.extract (A, { }, I) ;
        end
        if (whole_vector & size (C,1) == 1)
            C = C.' ;
        end
    end
elseif (ndims == 2)
    % C = A (I,J)
    I = get_index (S.subs (1)) ;
    J = get_index (S.subs (2)) ;
    C = gb.extract (A, I, J) ;
else
    error ('%dD indexing not supported', ndims) ;
end

