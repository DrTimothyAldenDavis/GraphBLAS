function C = eq (A, B)
%A == B Equal.
% C = (A == B) is an element-by-element comparison of A and B.  One or
% both may be scalars.  Otherwise, A and B must have the same size.
%
% The input matrices may be either GraphBLAS and/or MATLAB matrices, in
% any combination.  C is returned as a GraphBLAS matrix.

% The pattern of C depends on the type of inputs:
% A scalar, B scalar:  C is scalar.
% A scalar, B matrix:  C is full if A==0, otherwise C is a subset of B.
% B scalar, A matrix:  C is full if B==0, otherwise C is a subset of A.
% A matrix, B matrix:  C is full.
% Zeroes are then dropped from C after it is computed.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2019, All Rights Reserved.
% http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

if (isscalar (A))
    if (isscalar (B))
        % both A and B are scalars.  C is full.
        C = dense_comparator ('==', A, B) ;
    else
        % A is a scalar, B is a matrix
        if (get_scalar (A) == 0)
            % since a == 0, entries not present in B result in a true value,
            % so the result is dense.  Expand A to a dense matrix.
            A = gb.expand (A, true (size (B))) ;
            if (~gb.isfull (B))
                B = full (B) ;
            end
            C = gb.prune (gb.emult ('==', A, B)) ;
        else
            % since a ~= 0, entries not present in B result in a false
            % value, so the result is a sparse subset of B.  select all
            % entries in B == a, then convert to true.
            C = gb.apply ('1.logical', gb.select ('==thunk', B, A)) ;
        end
    end
else
    if (isscalar (B))
        % A is a matrix, B is a scalar
        if (get_scalar (B) == 0)
            % since b == 0, entries not present in A result in a true value,
            % so the result is dense.  Expand B to a dense matrix.
            B = gb.expand (B, true (size (A))) ;
            if (~gb.isfull (A))
                A = full (A) ;
            end
            C = gb.prune (gb.emult ('==', A, B)) ;
        else
            % since b ~= 0, entries not present in A result in a false
            % value, so the result is a sparse subset of A.  select all
            % entries in A == b, then convert to true.
            C = gb.apply ('1.logical', gb.select ('==thunk', A, B)) ;
        end
    else
        % both A and B are matrices.  C is full.
        C = dense_comparator ('==', A, B) ;
    end
end

