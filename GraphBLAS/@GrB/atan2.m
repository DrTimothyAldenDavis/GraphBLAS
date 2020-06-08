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

if (isobject (A))
    A = A.opaque ;
end

if (isobject (B))
    B = B.opaque ;
end

atype = gbtype (A) ;
btype = gbtype (B) ;

if (contains (atype, 'complex') || contains (btype, 'complex'))
    gb_error ('both inputs must be real') ;
end

if (~gb_isfloat (atype))
    A = gbnew (A, 'double') ;
    atype = 'double' ;
end

if (~gb_isfloat (btype))
    B = gbnew (B, 'double') ;
    btype = 'double' ;
end

% atan2(A,B) gives the set union of the pattern of A and B

ctype = gboptype (atype, btype) ;

if (gb_isscalar (A))
    if (gb_isscalar (B))
        % both A and B are scalars
        C = GrB (gbemult ('atan2', gbfull (A, ctype), gbfull (B, ctype))) ;
    else
        % A is a scalar, B is a matrix
        A = gb_expand (A, B, ctype) ;
        C = GrB (gbemult ('atan2', A, B)) ;
    end
else
    if (gb_isscalar (B))
        % A is a matrix, B is a scalar
        B = gb_expand (B, A, ctype) ;
        C = GrB (gbemult ('atan2', A, B)) ;
    else
        % both A and B are matrices.  C is the set union of A and B.
        C = GrB (gb_union_op ('atan2', A, B)) ;
    end
end

