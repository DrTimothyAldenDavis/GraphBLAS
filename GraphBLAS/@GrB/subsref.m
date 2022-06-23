function C = subsref (A, S)
%SUBSREF C = A(I,J) or C = A(I); extract submatrix.
% C = A(I,J) extracts the A(I,J) submatrix of the GraphBLAS matrix A.
% With a single index, C = A(I) extracts a subvector C of a vector A.
% For linear indexing of a 2D matrix, only C=A(:) is currently supported.
% C=A(I) is not yet supported if A is a 2D matrix.
%
% x = A (M) for a logical matrix M constructs an nnz(M)-by-1 vector x, for
% built-in-style logical indexing.  A or M may be built-in sparse or full
% matrices, or GraphBLAS matrices, in any combination.  M must be either a
% built-in logical matrix (sparse or full), or a GraphBLAS logical matrix;
% that is, GrB.type (M) must be 'logical'.
%
% GraphBLAS can construct huge sparse matrices, but they cannot always be
% indexed with A(lo:hi,lo:hi), because of a limitation of the built-in
% colon notation.  A colon expression is expanded into an explicit vector,
% but this can be too big.   Instead of the colon notation start:inc:fini,
% use a cell array with three integers, {start, inc, fini}.
%
% Example:
%
%   n = 1e14 ;
%   H = GrB (n, n)               % a huge empty matrix
%   I = [1 1e9 1e12 1e14] ;
%   M = magic (4)
%   H (I,I) = M
%   J = {1, 1e13} ;             % represents 1:1e13 colon notation
%   C = H (J, J)                % this is very fast
%   E = H (1:1e13, 1:1e13)      % but this is not possible
%
% See also GrB/subsasgn, GrB/subsindex, GrB.subassign, GrB.assign,
% GrB.extract.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2022, All Rights Reserved.
% SPDX-License-Identifier: GPL-3.0-or-later

% FUTURE: add linear indexing.

if (isobject (A))
    A = A.opaque ;
end
[m, n] = gbsize (A) ;

if (length (S) > 1)
    error ('nested indexing not supported') ;
end

if (~isequal (S.type, '()'))
    error ('index type %s not supported', S.type) ;
end

ndims = length (S.subs) ;

if (ndims == 1)

    % C = A(M) if M is logical, or C=A(I) otherwise
    S = S.subs {1} ;
    if (isobject (S))
        S = S.opaque ;
    end
    if (isequal (gbtype (S), 'logical'))
        % C = A (M) for logical indexing
        C = GrB (gblogextract (A, S)) ;
    else
        % C = A (I)
        [I, whole] = gb_index (S) ;
        if (m == 1 || n == 1)
            % C = A (I) for a vector A
            if (m > 1)
                C = gbextract (A, I, { }) ;
            else
                C = gbextract (A, { }, I) ;
            end
            [cm, ~] = gbsize (C) ;
            if (whole && cm == 1)
                C = gbtrans (C) ;
            end
            C = GrB (C) ;
        else
            % C = A (I) for a matrix A
            if (whole)
                % C = A (:), whole matrix case
                % FUTURE: C=A(:) would be faster with GxB_reshape
                desc.base = 'zero-based' ;
                [I0, J0, X] = gbextracttuples (A, desc) ;
                [I1, mn] = gb_2d_to_1d (I0, J0, m, n) ;
                J1 = int64 (0) ;
                clear I0 J0 ;
                C = GrB (gbbuild (I1, J1, X, mn, 1, desc)) ;
            else
                % C = A (I), general case not yet supported
                error ('Except for C=A(:), linear indexing not yet supported') ;
            end
        end
    end

elseif (ndims == 2)

    % C = A (I,J)
    C = GrB (gbextract (A, gb_index (S.subs {1}), gb_index (S.subs {2}))) ;

else

    % sparse N-dimensional arrays for N > 2 will not be supported
    error ('%dD indexing not supported', ndims) ;

end

