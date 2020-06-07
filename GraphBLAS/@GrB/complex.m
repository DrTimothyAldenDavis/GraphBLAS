function C = complex (A, B)
%COMPLEX cast a GraphBLAS matrix to MATLAB sparse double complex matrix.
% C = complex (G) typecasts the GraphBLAS matrix G to into a MATLAB sparse
% complex matrix.
%
% With two inputs, C = complex (A,B) returns a MATLAB matrix C = A + 1i*B,
% where A or B are real matrices (MATLAB and/or GraphBLAS, in any
% combination).  If A or B are nonzero scalars and the other input is a
% matrix, or if both A and B are scalars, C is full.  Otherwise, C is
% sparse.
%
% To typecast the matrix G to a GraphBLAS sparse double complex matrix
% instead, use C = GrB (G, 'complex') or C = GrB (G, 'double complex').
% To typecast the matrix G to a GraphBLAS single complex matrix, use
% C = GrB (G, 'single complex').
%
% To construct a complex GraphBLAS matrix from real GraphBLAS matrices
% A and B, use C = A + 1i*B instead.
%
% Since MATLAB does not support sparse single complex matrices, C is
% always returned as a double complex matrix (sparse or full).
%
% See also cast, GrB, GrB/double, GrB/single, GrB/logical, GrB/int8,
% GrB/int16, GrB/int32, GrB/int64, GrB/uint8, GrB/uint16, GrB/uint32,
% GrB/uint64.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights
% Reserved. http://suitesparse.com.  See GraphBLAS/Doc/License.txt.

if (nargin == 1)

    % with a single input, A must be a GraphBLAS matrix (otherwise,
    % this overloaded method for GrB objects would not be called).
    % Convert A to a double complex matrix C.
    C = gbsparse (A.opaque, 'double complex') ;

else

    if (isobject (A))
        A = A.opaque ;
    end

    if (isobject (B))
        B = B.opaque ;
    end

    if (contains (gbtype (A), 'complex') || contains (gbtype (B), 'complex'))
        error ('inputs must be real') ;
    end

    [am, an] = gbsize (A) ;
    [bm, bn] = gbsize (B) ;
    a_is_scalar = (am == 1) && (an == 1) ;
    b_is_scalar = (bm == 1) && (bn == 1) ;

    if (a_is_scalar)
        if (b_is_scalar)
            % both A and B are scalars.  C is also a scalar.
            A = gbfull (A, 'double') ;
            B = gbfull (B, 'double') ;
            desc.kind = 'full' ;
        else
            % A is a scalar, B is a matrix.  C is full, unless A == 0.
            if (gb_get_scalar (A) == 0)
                % C = 1i*B, so A = zero, expanded to the pattern of B;
                % C is sparse
                A = gb_expand (0, B, 'double') ;
                desc.kind = 'sparse' ;
            else
                % expand A and B to full double matrices; C is full
                A = gb_scalar_to_full (bm, bn, 'double', A) ;
                B = gbfull (B, 'double') ;
                desc.kind = 'full' ;
            end
        end
        % A and B now have the same pattern, so emult can be used
        C = gbemult ('cmplx.double', A, B, desc) ;
    else
        if (b_is_scalar)
            % A is a matrix, B is a scalar.  C is full, unless B == 0.
            if (gb_get_scalar (B) == 0)
                % C = complex (A); C is sparse
                C = gbsparse (A, 'double.complex') ;
            else
                % expand A and B to full double matrices; C is full
                A = gbfull (A, 'double') ;
                B = gb_scalar_to_full (am, an, 'double', B) ;
                desc.kind = 'full' ;
                C = gbemult ('cmplx.double', A, B, desc) ;
            end
        else
            % both A and B are matrices.  C is sparse.
            desc.kind = 'sparse' ;
            C = gb_union_op ('cmplx.double', A, B, desc) ;
        end
    end

end

