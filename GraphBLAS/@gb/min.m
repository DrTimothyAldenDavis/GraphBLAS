function C = min (varargin)
%MIN Minimum elements of a GraphBLAS or MATLAB matrix.
%
% C = min (G) is the smallest entry in the vector G.  If G is a matrix,
% C is a row vector with C(j) = min (G (:,j)).
%
% C = min (A,B) is an array of the element-wise minimum of two matrices
% A and B, which either have the same size, or one can be a scalar.
% Either A and/or B can be GraphBLAS or MATLAB matrices.
%
% C = min (G, [ ], 'all') is a scalar, with the smallest entry in G.
% C = min (G, [ ], 1) is a row vector with C(j) = min (G (:,j))
% C = min (G, [ ], 2) is a column vector with C(i) = min (G (i,:))
%
% The indices of the minimum entry, or [C,I] = min (...) in the MATLAB
% built-in min function, are not computed.  The min (..., nanflag) option
% is not available; only the 'includenan' behavior is supported.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

G = varargin {1} ;
[m n] = size (G) ;

if (isequal (gb.type (G), 'logical'))
    op = '&.logical' ;
else
    op = 'min' ;
end

if (nargin == 1)

    % C = min (G)
    if (isvector (G))
        % C = min (G) for a vector G results in a scalar C
        C = gb.reduce (op, G) ;
        if (gb.nvals (G) < m*n) ;
            C = min (C, 0) ;
        end
    else
        % C = min (G) reduces each column to a scalar,
        % giving a 1-by-n row vector.
        C = gb.vreduce (op, G, struct ('in0', 'transpose')) ;
        % if C(j) > 0, but the column is sparse, then assign C(j) = 0.
        C = gb.subassign (C, (C > 0) & (col_degree (G) < m), 0)' ;
    end

elseif (nargin == 2)

    % C = min (A,B)
    A = varargin {1} ;
    B = varargin {2} ;
    if (isscalar (A))
        if (isscalar (B))
            % both A and B are scalars.  Result is also a scalar.
            C = sparse_comparator (op, A, B) ;
        else
            % A is a scalar, B is a matrix
            if (get_scalar (A) < 0)
                % since A < 0, the result is full
                C = gb.eadd (op, gb.expand (A, true (size (B))), B) ;
            else
                % since A >= 0, the result is sparse.  Expand the scalar A
                % to the pattern of B.
                C = gb.eadd (op, gb.expand (A, B), B) ;
            end
        end
    else
        if (isscalar (B))
            % A is a matrix, B is a scalar
            if (get_scalar (B) < 0)
                % since B < 0, the result is full
                C = gb.eadd (op, A, gb.expand (B, true (size (A)))) ;
            else
                % since B >= 0, the result is sparse.  Expand the scalar B
                % to the pattern of A.
                C = gb.eadd (op, A, gb.expand (B, A)) ;
            end
        else
            % both A and B are matrices.  Result is sparse.
            C = sparse_comparator (op, A, B) ;
        end
    end

elseif (nargin == 3)

    % C = min (G, [ ], option)
    option = varargin {3} ;
    if (isequal (option, 'all'))
        % C = min (G, [ ] 'all'), reducing all entries to a scalar
        C = gb.reduce (op, G) ;
        if (gb.nvals (G) < m*n) ;
            C = min (C, 0) ;
        end
    elseif (isequal (option, 1))
        % C = min (G, [ ], 1) reduces each column to a scalar,
        % giving a 1-by-n row vector.
        C = gb.vreduce (op, G, struct ('in0', 'transpose')) ;
        % if C(j) > 0, but the column is sparse, then assign C(j) = 0.
        C = gb.subassign (C, (C > 0) & (col_degree (G) < m), 0)' ;
    elseif (isequal (option, 2))
        % C = min (G, [ ], 2) reduces each row to a scalar,
        % giving an m-by-1 column vector.
        C = gb.vreduce (op, G) ;
        % if C(i) > 0, but the row is sparse, then assign C(i) = 0.
        C = gb.subassign (C, (C > 0) & (row_degree (G) < n), 0) ;
    else
        error ('unknown option') ;
    end

else
    error ('invalid usage') ;
end

