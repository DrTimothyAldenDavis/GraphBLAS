function C = atan2 (A, B)
%ATAN2 four quadrant inverse tangent.
% C = atan2 (X,Y) is the 4 quadrant arctangent of the entries in the
% GraphBLAS matrices X and Y.  The input matrices X and Y may be either
% GraphBLAS and/or MATLAB matrices, in any combination.  C is returned as a
% GraphBLAS matrix.
%
% See also GrB/tan, GrB/tanh, GrB/atan, GrB/atanh.

% SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights
% Reserved. http://suitesparse.com.  See GraphBLAS/Doc/License.txt.

if (~(isreal (A) && isreal (B)))
    error ('both inputs must be real') ;
end

if (~isfloat (A))
    A = GrB (A, 'double') ;
end

if (~isfloat (B))
    B = GrB (B, 'double') ;
end

% atan2(A,B) gives the set union of the pattern of A and B

ctype = GrB.optype (A, B) ;

if (isscalar (A))
    if (isscalar (B))
        % both A and B are scalars
        C = GrB.emult ('atan2', gb_full (A, ctype), gb_full (B, ctype)) ;
    else
        % A is a scalar, B is a matrix
        A = GrB.expand (GrB (A, ctype), B) ;
        C = GrB.emult ('atan2', A, B) ;
    end
else
    if (isscalar (B))
        % A is a matrix, B is a scalar
        B = GrB.expand (GrB (B, ctype), A) ;
        C = GrB.emult ('atan2', A, B) ;
    else
        % both A and B are matrices.  C is the set union of A and B.
        C = gb_union_op ('atan2', A, B) ;
    end
end

