function C = min (varargin)
%MIN Minimum elements of a GraphBLAS or MATLAB matrix.
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
% The 2nd output of [C,I] = min (...) in the MATLAB built-in min
% is not yet supported.  The min (..., nanflag) option is
% not yet supported; only the 'omitnan' behavior is supported.
%
% Complex matrices are not supported.
%
% See also GrB/max.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights
% Reserved. http://suitesparse.com.  See GraphBLAS/Doc/License.txt.

G = varargin {1} ;
type = GrB.type (G) ;
if (contains (type, 'complex'))
    error ('complex matrices not supported') ;
elseif (isequal (type, 'logical'))
    op = '&.logical' ;
else
    op = 'min' ;
end

[m, n] = size (G) ;
desc = struct ('in0', 'transpose') ;

if (nargin == 1)

    % C = min (G)
    if (isvector (G))
        % C = min (G) for a vector G results in a scalar C
        C = GrB.reduce (op, G) ;
        if (~GrB.isfull (G))
            C = min (C, 0) ;    % recursively, on a scalar
        end
    else
        % C = min (G) reduces each column to a scalar,
        % giving a 1-by-n row vector.
        C = GrB.vreduce (op, G, desc) ;
        % if C(j) > 0, but the column is sparse, then assign C(j) = 0.
        coldegree = GrB.entries (G, 'col', 'degree') ;
        C = GrB.subassign (C, (C > 0) & (coldegree < m), 0)' ;
    end

elseif (nargin == 2)

    % C = min (A,B)
    A = varargin {1} ;
    B = varargin {2} ;
    ctype = GrB.optype (A, B) ;
    if (isscalar (A))
        if (isscalar (B))
            % both A and B are scalars.  Result is also a scalar.
            C = GrB (gb_union_op (op, gb (A), gb (B))) ;
        else
            % A is a scalar, B is a matrix
            if (gb_get_scalar (A) < 0)
                % since A < 0, the result is full
                [m, n] = size (B) ;
                % A (1:m,1:n) = A and cast to ctype
                A = GrB.subassign (GrB (m, n, ctype), A) ;
            else
                % since A >= 0, the result is sparse.  Expand the scalar A
                % to the pattern of B.
                A = GrB.expand (GrB (A, ctype), B) ;
            end
            C = GrB.eadd (A, op, B) ;
        end
    else
        if (isscalar (B))
            % A is a matrix, B is a scalar
            if (gb_get_scalar (B) < 0)
                % since B < 0, the result is full
                [m, n] = size (A) ;
                % B (1:m,1:n) = B and cast to ctype
                B = GrB.subassign (GrB (m, n, ctype), B) ;
            else
                % since B >= 0, the result is sparse.  Expand the scalar B
                % to the pattern of A.
                B = GrB.expand (GrB (B, ctype), A) ;
            end
            C = GrB.eadd (A, op, B) ;
        else
            % both A and B are matrices.  Result is sparse.
            C = GrB (gb_union_op (op, gb (A), gb (B))) ;
        end
    end

elseif (nargin == 3)

    % C = min (G, [ ], option)
    option = varargin {3} ;
    if (isequal (option, 'all'))
        % C = min (G, [ ] 'all'), reducing all entries to a scalar
        C = GrB.reduce (op, G) ;
        if (~GrB.isfull (G))
            C = min (C, 0) ;
        end
    elseif (isequal (option, 1))
        % C = min (G, [ ], 1) reduces each column to a scalar,
        % giving a 1-by-n row vector.
        C = GrB.vreduce (op, G, desc) ;
        % if C(j) > 0, but the column is sparse, then assign C(j) = 0.
        coldegree = GrB.entries (G, 'col', 'degree') ;
        C = GrB.subassign (C, (C > 0) & (coldegree < m), 0)' ;
    elseif (isequal (option, 2))
        % C = min (G, [ ], 2) reduces each row to a scalar,
        % giving an m-by-1 column vector.
        C = GrB.vreduce (op, G) ;
        % if C(i) > 0, but the row is sparse, then assign C(i) = 0.
        rowdegree = GrB.entries (G, 'row', 'degree') ;
        C = GrB.subassign (C, (C > 0) & (rowdegree < n), 0) ;
    else
        gb_error ('unknown option') ;
    end

else
    gb_error ('invalid usage') ;
end

